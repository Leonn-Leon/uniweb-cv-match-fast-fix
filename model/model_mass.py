import os
import random
import requests
import re
from ast import literal_eval
from copy import deepcopy
from datetime import datetime, timedelta, date
from typing import Dict, List, Union

import nltk
import numpy as np
import pandas as pd
import simplemma
import yaml
from dotenv import load_dotenv
from geopy.distance import geodesic
# from geopy.geocoders import Nominatim
# from geopy.location import Location
from loguru import logger
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from openai import OpenAI
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
import functools

from model import CvSelector
from utils.enums import Method, ModeInfo
from utils.request_api import process_corpus

load_dotenv()
nltk.download("punkt_tab")


class MassCvSelector(CvSelector):
    def __init__(
        self, config: Dict, api_token: str, method: Method = Method.EMBEDDINGS
    ):
        CvSelector.__init__(
            self,
            config=config,
            api_token=api_token,
            method=method,
        )
        self.stemmer = SnowballStemmer(language="russian")
        self.days_filter_threshold = (datetime.now().date() - date.fromisoformat(config["stage_1"]["date_threshold"]) ).days
        self.first_filter = config["stage_1"]["first_filter"]
        self.top_n_init = config["stage_1"]["top_n_init"]
        self.close_dist_threshold = config["stage_1"]["close_dist_threshold"]
        # self.score_threshold_stage_2 = config["stage_2"]["score_threshold"]

    def __normalize_text(self, text: str):
        text = text.lower().replace("\n", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join(sorted(text.split(" ")))
        return text

    def __minmax_scale(self, x: np.ndarray):
        range_ = x.max() - x.min()
        if range_ == 0:
            return np.zeros_like(x)
        return (x - x.min()) / range_

    def __filter_1_stage(
        self, date: date, availability: str, days_thresh: int
    ):
        if datetime.now().date() - timedelta(days=days_thresh) > date:
            return False
        if availability == "Нет":
            return False
        return True

    def __score_move(self, move: str, dist: float, dist_thresh: float = 50):
        if dist < dist_thresh:
            return 1.0
        value_array = np.array(["Невозможен", "Нет данных", "Возможен"])
        return float(np.where(value_array == move)[0][0]) / 2

    def __shorten_address(self, address: str):
        split_ = address.split(", ")
        split_ = split_[:3]
        result = []
        for item in split_:
            norm = True
            key_words = ["ул.", "улиц", "пер.", "переулок", "пл.", "площ"]
            for key in key_words:
                if key in item:
                    norm = False
                    break
            if norm:
                result.append(item)
        return ", ".join(result)

    def match_job(self, position_vac: str, positions_condidates: List[str]):
        client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
        prompt = f"Выберите подходящую должность.\nДолжность из вакансии: '{position_vac}'\nДолжности кандидатов: {str(positions_condidates)}"
        system_prompt = """Вы эксперт в найме персонала. Ваша задача для должности из вакансии выбрать
        самую подходящую должность из списка должностей кандидатов. Верните в ответ только выбранную должность из списка. Не нужно
        выводить никаких объяснений. Должность из списка нужно вернуть точно, без каких-либо изменений.
        """
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ], temperature=0.01
        )
        return completion.choices[0].message.content

    def find_info(self, info: str):
        client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
        description, mode, query, query_desc, delete_data = info.split("[SEP]")
        # logger.info(f"find_info: {info}")

        # 1. Формируем базовый системный промпт
        system_prompt = f"""Ты - ассистент по извлечению информации из текста. Твоя задача - найти в предоставленном тексте ('Описание') ответ на вопрос пользователя.
            Верни результат в формате JSON с одним ключом '{query}'.
            Если информация найдена, помести ее в значение ключа '{query}'.
            Если информация не найдена, используй значение "Нет данных". Не придумывай информацию."""

        # 2. Добавляем инструкцию про удаление данных, ТОЛЬКО если delete_data передан
        # if delete_data is not None and delete_data != "None":
        #     system_prompt += f"\nВАЖНО: Из найденной информации исключи любые данные, которые пересекаются с этим списком для удаления: {delete_data}"

        # 3. Формируем пользовательский промпт: Описание + Вопрос из query_desc
        # Используем query_desc как основной вопрос. Если его нет, используем query как запасной вариант.
        question = query_desc if query_desc is not None and query_desc != "None" else f"Извлеки данные для категории '{query}'."

        # Собираем пользовательский промпт
        user_prompt = f"Описание:\n{description}\n\nВопрос:\n{question}\n\nОтвет (в формате JSON):"

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],temperature=0.01,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.1))
    def get_coords(self, address_search: str):
        """
        Получает координаты и уточненный адрес с помощью API Яндекс Геокодера.
        """
        api_key = os.getenv("YANDEX_GEOCODER_API_KEY")
        if not api_key:
            logger.error("Yandex Geocoder API key not found in environment variables.")
            return None, address_search # Возвращаем None и исходный адрес при ошибке

        if not address_search or pd.isna(address_search): # Проверка на пустой или NaN адрес
            logger.warning(f"Attempted to geocode empty or NaN address.")
            return None, address_search

        base_url = "https://geocode-maps.yandex.ru/v1/"
        params = {
            "apikey": api_key,
            "geocode": address_search,
            "format": "json",
            "results": 1 # Запрашиваем только один, самый релевантный результат
        }

        try:
            response = requests.get(base_url, params=params, timeout=100)
            response.raise_for_status() # Проверка на HTTP ошибки (4xx, 5xx)
            data = response.json()

            feature_member = data.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
            if not feature_member:
                logger.warning(f"No geocode result found for address: {address_search}")
                return None, address_search

            # Извлекаем координаты и адрес
            geo_object = feature_member[0].get("GeoObject", {})
            point_str = geo_object.get("Point", {}).get("pos")
            formatted_address = geo_object.get("metaDataProperty", {}).get("GeocoderMetaData", {}).get("Address", {}).get("formatted", address_search)
            precision = geo_object.get("metaDataProperty", {}).get("GeocoderMetaData", {}).get("precision", "unknown")

            if point_str:
                # Яндекс возвращает "долгота широта"
                lon, lat = map(float, point_str.split())
                logger.info(f"Geocoded '{address_search}' to ({lat}, {lon}) with precision '{precision}'. Formatted: '{formatted_address}'")
                # Возвращаем в формате (широта, долгота) как ожидает geopy
                return (lat, lon), formatted_address
            else:
                logger.warning(f"Could not extract coordinates for address: {address_search}")
                return None, formatted_address # Возвращаем None для координат, но уточненный адрес

        except requests.exceptions.RequestException as e:
            logger.error(f"Yandex Geocoder request failed for '{address_search}': {e}")
            return None, address_search
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse Yandex Geocoder response for '{address_search}': {e}")
            return None, address_search
        except Exception as e:
            logger.error(f"An unexpected error occurred during geocoding for '{address_search}': {e}")
            return None, address_search


    def preprocess_vacancy(self, vacancy: Dict):
        for cat in tqdm(["Адрес", "График работы", "Тип занятости", "Переезд"]):
            info = "[SEP]".join(
                [
                    vacancy["location"],
                    ModeInfo.VACANCY,
                    cat,
                    self.category_desc.get(cat, "None"),
                    "None",
                ]
            )
            info_extracted = self.find_info(info=info)
            info_dict = self.postprocess_extracted_info(info=info_extracted, cat=cat)
            info_dict[cat.capitalize()] = info_dict.pop(cat)
            vacancy.update(info_dict)

        for cat in tqdm(
            [
                "Навыки",
                "опыт работы",
                "образование",
                "категория прав",
            ]
        ):
            info = "[SEP]".join(
                [
                    vacancy["required"] + "\n" + vacancy["additional"],# Откуда ищем инфу
                    ModeInfo.VACANCY,
                    cat,
                    self.category_desc.get(cat, "None"),
                    "None",
                ]
            )
            info_extracted = self.find_info(info=info)
            info_dict = self.postprocess_extracted_info(info=info_extracted, cat=cat)
            info_dict[cat.capitalize()] = info_dict.pop(cat)
            vacancy.update(info_dict)

        vacancy["Описание"] = (
            vacancy["required"]
            + "\n"
            + vacancy["additional"]
            + "\n"
            + vacancy["location"]
        )
        vacancy["Full_description"] = self.get_desc(
            vacancy=vacancy, keys=self.keys_vacancy
        )
        vacancy["Адрес"] = self.__shorten_address(vacancy["Адрес"])
        logger.info(f"Geo search failed: {vacancy['Адрес']}")
        try:
            vacancy["coords"], vacancy["Адрес"] = self.get_coords(vacancy["Адрес"])
        except Exception:
            vacancy["coords"] = None
        logger.info(f"Coords: {vacancy['coords']}")
        return vacancy

    def normalize_address(self, address: str):
        address = address.lower()
        address = re.sub(r"[^\w\s]", "", address)
        address = re.sub(r"\d+", "", address)
        address = re.sub(r"\s+", " ", address).strip()
        address = " ".join(sorted(address.split(" ")))
        return address

    def compare_addresses(self, addr1, addr2):
        similarity = fuzz.ratio(addr1, addr2, processor=self.normalize_address)
        return 100 - similarity

    def __tokenize_feat(self, text: str):
        norm_text = self.__normalize_text(text)
        words = word_tokenize(norm_text, language="russian")
        prep_words = []
        for word in words:
            word_lemm = simplemma.lemmatize(word, lang="ru")
            word_stemm = self.stemmer.stem(word_lemm)
            prep_words.append(word_stemm)
        return prep_words

    def __bm25_score(self, feature_list: List[str], query: str):
        tokenized_corpus = [self.__tokenize_feat(x) for x in feature_list]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.__tokenize_feat(query)
        scores = np.abs(bm25.get_scores(tokenized_query))
        return self.__minmax_scale(scores)

    def find_geo_distance(
        self, coords_vac: Union[str, tuple], coords_cand: Union[str, tuple]
    ):
        if type(coords_vac) == str:
            coords_vac = literal_eval(coords_vac)
        if type(coords_cand) == str:
            coords_cand = literal_eval(coords_cand)
        return geodesic(coords_vac, coords_cand).kilometers

    def distance_score(self, distances: np.ndarray) -> np.ndarray:
        conditions = [
            distances <= 100,
            distances <= 500,
            distances <= 1000,
        ]
        scores = [
            1.0,
            0.4,
            0.1,
        ]
        default_score = 0.0
        result_scores = np.select(conditions, scores, default=default_score)
        return result_scores

    def rank_first_stage(self, vacancy: Dict, df_relevant: pd.DataFrame,
                         date_threshold: datetime.date, is_vahta: bool, max_distance_filter: float):
        """
        Первый этап ранжирования:
        1. Фильтрует дубликаты, обрабатывает даты/адреса.
        2. Геокодирует, рассчитывает расстояние и схожесть по переезду.
        3. Фильтрует по готовности к переезду/близости.
        4. Фильтрует по дате обновления резюме и доступности.
        5. Сортирует по расстоянию и выбирает top_n_init самых близких.
        6. Фильтрует выбранных кандидатов по должности (BM25 -> Embeddings).
        7. Рассчитывает остальные схожести (расстояние(норм), дата) для оставшихся.
        8. Считает итоговый балл sim_score_first.
        9. Возвращает top_n_first_stage лучших по sim_score_first.
        """

        self.days_filter_threshold = (datetime.now().date() - date_threshold).days
        print(f"days_filter_threshold: {self.days_filter_threshold}")

        ## 1. Начальная подготовка
        df_relevant = df_relevant.drop_duplicates(subset=["link"])
        logger.debug(f"Shape after drop_duplicates: {df_relevant.shape}")

        if df_relevant.empty:
             logger.warning("Initial df_relevant is empty. Returning empty DataFrame.")
             return pd.DataFrame()

        # Имена колонок из конфига для читаемости
        date_col = self.cluster_match_features[2]
        address_col = self.text_features[1] # address_short
        move_col = self.cluster_match_features[0] # Переезд
        distance_sim_col = self.cluster_match_features[1] # Схожесть по расстоянию

        # Преобразование даты и удаление NaN
        df_relevant[date_col] = pd.to_datetime(
            df_relevant[date_col], errors='coerce', yearfirst=True, format='mixed'
        )

        ## 4. Фильтр по дате и доступности
        logger.info("Filtering by date and availability...")

        if 'Доступность' not in df_relevant.columns:
             logger.warning("Column 'Доступность' not found. Skipping availability filter.")
             df_relevant_filtered = df_relevant.copy()
        else:
             df_relevant.loc[:, "filter"] = df_relevant[
                 [date_col, "Доступность"]
             ].apply(
                 lambda x: self.__filter_1_stage(
                     x.iloc[0].date(), x.iloc[1], self.days_filter_threshold
                 ),
                 axis=1,
             )
             df_relevant_filtered = df_relevant[df_relevant["filter"]].copy()
             logger.debug(f"Shape after date/availability filter: {df_relevant_filtered.shape}")

        # Если после фильтров по дате/доступности никого не осталось
        if df_relevant_filtered.empty:
             logger.warning("No candidates left after date/availability filters. Returning empty DataFrame.")
             expected_cols = df_relevant.columns.tolist() + ['sim_score_first']
             return pd.DataFrame(columns=list(set(expected_cols)))
        
        df_relevant = df_relevant_filtered.copy()

        # Удаляем строки, где нет валидной даты или адреса для геокодинга
        cols_to_dropna = [date_col]
        if address_col in df_relevant.columns:
            cols_to_dropna.append(address_col)
        else:
            logger.warning(f"Address column '{address_col}' not found for dropna step.")
        df_relevant.dropna(subset=cols_to_dropna, inplace=True)
        logger.debug(f"Shape after dropna({cols_to_dropna}): {df_relevant.shape}")

        # Если после dropna никого не осталось
        if df_relevant.empty:
             logger.warning("No candidates left after dropna. Returning empty DataFrame.")
             return pd.DataFrame()

        ## 2. Гео-обработка и расчет расстояний
        logger.info(f"Starting parallel coordinate processing for {len(df_relevant)} candidates...")

        if 'coords' not in df_relevant.columns:
            df_relevant['coords'] = None

        logger.info(f"Starting parallel processing of coordinates and distances for {len(df_relevant)} candidates using process_corpus...")

        vacancy_coords = vacancy.get("coords")
        vacancy_address_for_compare = vacancy.get(address_col, "")

        corpus_for_processing = []
        for index, row in df_relevant.iterrows():
             coords_val = row.get('coords')
             candidate_address = row.get(address_col, '')
             corpus_for_processing.append((coords_val, candidate_address))

        partial_worker = functools.partial(
            self._process_coords_and_distance,
            vacancy_coords=vacancy_coords,
            vacancy_address=vacancy_address_for_compare
        )

        logger.info(f"Calling process_corpus with {len(corpus_for_processing)} items for combined processing...")
        try:
            results_list = process_corpus(
                corpus=corpus_for_processing,
                func=partial_worker,
                num_workers=self.request_num_workers
            )
            logger.info("process_corpus for combined processing finished.")
        except Exception as e:
            logger.error(f"Error during process_corpus execution for combined processing: {e}")
            results_list = [(None, corpus_for_processing[i][1], np.nan) for i in range(len(corpus_for_processing))]

        results_coords = [item[0] if isinstance(item, tuple) and len(item) == 3 else None for item in results_list]
        results_addresses = [item[1] if isinstance(item, tuple) and len(item) == 3 else corpus_for_processing[i][1] for i, item in enumerate(results_list)]
        results_distances = [item[2] if isinstance(item, tuple) and len(item) == 3 else np.nan for item in results_list]

        df_relevant['coords'] = results_coords
        if address_col in df_relevant.columns:
             df_relevant[address_col] = results_addresses
        df_relevant['distance'] = pd.to_numeric(results_distances, errors='coerce')
        logger.info("DataFrame updated with processed coordinates, addresses, and distances.")

        median_distance = df_relevant["distance"].median()
        logger.info(f"Filling {df_relevant['distance'].isna().sum()} NaN distances with median: {median_distance:.2f}")
        df_relevant["distance"] = df_relevant["distance"].fillna(median_distance)

        ## 3. Расчет схожести по переезду и фильтрация
        logger.info("Calculating move similarity and filtering...")

        if move_col not in df_relevant.columns:
            logger.warning(f"Move column '{move_col}' not found. Skipping move filter.")
        else:
            df_relevant.loc[:, f"{move_col}_sim"] = (
                 df_relevant[[move_col, "distance"]]
                 .apply(
                    lambda x: self.__score_move(
                        x.iloc[0], x.iloc[1], self.close_dist_threshold
                    ),
                    axis=1,
                )
            )
             
        
        df_relevant_filtered = df_relevant.copy()

        ## 5. Сортировка по расстоянию и выбор self.first_filter
        logger.debug("Sorting candidates by distance (ascending)...")
        df_relevant_filtered = df_relevant_filtered.sort_values("distance", ascending=True)

        ########################## ФИЛЬТР ПО РАССТОЯНИЮ ##########################
        if max_distance_filter is not None:
            # Есть расстояние
            df_relevant_filtered = df_relevant_filtered[df_relevant_filtered["distance"] <= max_distance_filter].copy()
        
        if not is_vahta:
            df_relevant_filtered = df_relevant_filtered.head(self.first_filter).copy()
            logger.info(f"Selected top {len(df_relevant_filtered)} candidates based on distance (self.first_filter={self.first_filter}).")
        else:
            logger.info("Vahta mode is enabled. Skipping distance filter.")

        ## 6. Фильтрация по должности (BM25 -> Embeddings)
        position_col_name = "Должность"
        vacancy_job_title = vacancy.get(position_col_name, "")

        df_relevant_for_position_filter = df_relevant_filtered # Работаем с отобранными по расстоянию

        if not vacancy_job_title:
            logger.warning(f"Vacancy job title ('{position_col_name}') is empty. Skipping position filtering.")
            df_filtered_final = df_relevant_for_position_filter.copy() # Пропускаем фильтр
        elif position_col_name not in df_relevant_for_position_filter.columns:
            logger.warning(f"Column '{position_col_name}' not found in df_relevant_for_position_filter. Skipping position filtering.")
            df_filtered_final = df_relevant_for_position_filter.copy() # Пропускаем фильтр
        else:
            logger.info(f"Starting position filtering for '{vacancy_job_title}' on {len(df_relevant_for_position_filter)} candidates...")
            # Шаг 1: BM25 до self.top_n_init
            candidate_job_titles_bm25 = df_relevant_for_position_filter[position_col_name].astype(str).fillna("").to_list()
            # logger.debug(f"Calculating BM25 scores for job titles... {candidate_job_titles_bm25}")
            # Проверка на пустой список перед BM25
            if not candidate_job_titles_bm25:
                 logger.warning("Candidate job title list for BM25 is empty. Skipping BM25 step.")
                 df_top_bm25 = df_relevant_for_position_filter.copy() # Пропускаем BM25
            else:
                 bm25_scores = self.__bm25_score(
                     feature_list=candidate_job_titles_bm25,
                     query=vacancy_job_title,
                 )
                 df_relevant_for_position_filter["_tmp_bm25_sim"] = bm25_scores
                 # Ограничиваем self.top_n_init размером текущего DataFrame
                 current_top_bm25 = min(self.top_n_init, len(df_relevant_for_position_filter))
                 df_top_bm25 = df_relevant_for_position_filter.nlargest(current_top_bm25, "_tmp_bm25_sim").copy()
                 logger.info(f"Selected top {len(df_top_bm25)} candidates based on BM25 job title score.")

            if df_top_bm25.empty:
                logger.warning("No candidates left after BM25 job title filter.")
                # Возвращаем пустой DataFrame с ожидаемыми колонками
                expected_cols = df_relevant_for_position_filter.columns.tolist() + ['sim_score_first']
                if '_tmp_bm25_sim' in expected_cols: expected_cols.remove('_tmp_bm25_sim')
                return pd.DataFrame(columns=list(set(expected_cols)))

            # Шаг 2: Embeddings до TOP_EMBED
            logger.debug("Calculating embedding similarities for top BM25 job titles...")
            candidate_job_titles_emb = df_top_bm25[position_col_name].astype(str).fillna("Нет данных").to_list()

            # Проверка на пустой список перед эмбеддингами
            if not candidate_job_titles_emb:
                 logger.warning("Candidate job title list for Embeddings is empty. Skipping Embedding step.")
                 df_filtered_final = df_top_bm25.drop(columns=["_tmp_bm25_sim"], errors='ignore') # Используем результат BM25
            else:
                try:
                    embeddings_cand_list = self.embedder.embed_corpus(candidate_job_titles_emb)
                    if not embeddings_cand_list: raise ValueError("Embeddings list (candidates) is empty.")
                    embeddings_np = np.array(embeddings_cand_list)

                    embedding_vac_list = self.embedder.embed_corpus([vacancy_job_title])
                    if not embedding_vac_list: raise ValueError("Embeddings list (vacancy) is empty.")
                    embedding_vac_np = np.array(embedding_vac_list)

                    cos_sims = cosine_similarity(embedding_vac_np, embeddings_np)[0]
                    df_top_bm25["embedding_sim"] = cos_sims

                    # Ограничиваем TOP_EMBED размером текущего DataFrame
                    #  current_top_embed = min(TOP_EMBED, len(df_top_bm25))
                    #  df_filtered_final = df_top_bm25.nlargest(current_top_embed, "embed_sim").copy()
                    #  logger.info(f"Selected top {len(df_filtered_final)} candidates based on embedding job title score.")

                    df_filtered_final = df_top_bm25.drop(columns=["_tmp_bm25_sim"], errors='ignore')

                except Exception as e:
                    logger.error(f"Error during embedding similarity calculation for job titles: {e}. Skipping embedding filter.")
                    df_filtered_final = df_top_bm25.drop(columns=["_tmp_bm25_sim"], errors='ignore') # Используем результат BM25

        # Если после финальной фильтрации по должности никого не осталось
        if df_filtered_final.empty:
             logger.warning("No candidates left after final job title filtering (BM25 + Embeddings).")
             return pd.DataFrame(columns=df_filtered_final.columns.tolist() + ['sim_score_first'])

        ## 7. Расчет остальных схожестей для финального набора кандидатов (df_filtered_final)
        logger.info(f"Calculating final similarities for {len(df_filtered_final)} candidates...")

        # Рассчитываем схожесть по расстоянию для конечного скора
        dist_values = df_filtered_final["distance"].values
        df_filtered_final[f"{distance_sim_col}_sim"] = self.distance_score(dist_values)
        logger.debug("Calculated normalized distance similarity.")

        # Рассчитываем схожесть по дате обновления резюме
        df_filtered_final[f"{date_col}_sim"] = (
            1 - df_filtered_final[date_col].apply(
                lambda x: (datetime.now().date() - x.date()).days
            )
            / self.days_filter_threshold
        ).clip(0, 1)
        logger.debug("Calculated date similarity.")

        ## 8. Расчет итогового балла первого этапа для df_filtered_final
        features_rank = [
            f"{feat}_sim" for feat in self.cluster_match_features # move_sim, distance_sim, date_sim
        ]
        missing_sim_cols = [col for col in features_rank if col not in df_filtered_final.columns]
        if missing_sim_cols:
            logger.error(f"Missing similarity columns for final score calculation: {missing_sim_cols}")
            for col in missing_sim_cols:
                df_filtered_final[col] = 0.0 # Присваиваем 0, чтобы избежать ошибки

        total_weight = self.first_stage_weights.sum()
        if total_weight == 0:
            logger.warning("Total weight for first stage score is zero.")
            df_filtered_final["sim_score_first"] = 0.0
        else:
            try:
                 sim_matrix = df_filtered_final[features_rank].values
                 df_filtered_final["sim_score_first"] = (
                     np.dot(sim_matrix, self.first_stage_weights)
                     / total_weight
                 )
                 logger.info("Calculated final sim_score_first.")
            except Exception as e:
                 logger.error(f"Error calculating final sim_score_first: {e}")
                 df_filtered_final["sim_score_first"] = 0.0

        ## 9. Финальная сортировка и возврат top_n_first_stage
        df_ranked = df_filtered_final.sort_values("sim_score_first", ascending=False)
        logger.info(f"Returning top {self.top_n_first_stage} candidates from stage 1.")
        return df_ranked.head(self.top_n_first_stage)

    def _process_coords_and_distance(self, candidate_data, vacancy_coords, vacancy_address):
        """
        Определяет координаты кандидата и считает расстояние до вакансии.

        Args:
            candidate_data (tuple): Данные кандидата (coords_input, candidate_address).
            vacancy_coords (tuple or None): Координаты вакансии.
            vacancy_address (str): Адрес вакансии для сравнения строк.

        Returns:
            tuple: (координаты_кандидата, адрес_кандидата, расстояние_до_вакансии).
        """
        coords_input, candidate_address = candidate_data # Распаковываем входные данные
        processed_coords = None

        # 1. Пытаемся получить координаты из входных данных (строка или готовый кортеж)
        if isinstance(coords_input, str) and coords_input.strip():
            # Пробуем распарсить строку типа "(lat, lon)"
            try:
                potential_coords = literal_eval(coords_input)
                if isinstance(potential_coords, (tuple, list)) and len(potential_coords) == 2:
                    lat = float(potential_coords[0])
                    lon = float(potential_coords[1])
                    processed_coords = (lat, lon)
            except Exception: pass # Ошибки парсинга игнорируем, перейдем к геокодингу
        elif isinstance(coords_input, (tuple, list)) and len(coords_input) == 2:
            # Проверяем, если на входе уже был кортеж/список с числами
            try:
                lat = float(coords_input[0])
                lon = float(coords_input[1])
                processed_coords = (lat, lon)
            except (ValueError, TypeError): pass # Не числа, перейдем к геокодингу

        # 2. Если координаты не найдены, пробуем геокодировать по адресу
        if processed_coords is None:
            # Используем адрес кандидата для геокодинга
            if pd.notna(candidate_address) and str(candidate_address).strip():
                try:
                    # Вызываем наш метод геокодинга через Яндекс API
                    cand_coords, _ = self.get_coords(candidate_address) # Форматированный адрес нам тут не нужен
                    if cand_coords:
                        processed_coords = cand_coords # Сохраняем найденные координаты
                except Exception as e:
                    # Ошибка при вызове геокодера
                    logger.debug(f"Geocoding failed for '{candidate_address}' in worker: {e}")
            # Если адреса нет или геокодинг не удался, processed_coords останется None

        # 3. Считаем расстояние до вакансии
        distance = np.nan # По умолчанию расстояние не известно
        if vacancy_coords is not None:
            # Если есть координаты вакансии, приоритет у гео-расстояния
            if isinstance(processed_coords, (tuple, list)) and len(processed_coords) == 2:
                try:
                    # Считаем расстояние между точками
                    distance = self.find_geo_distance(vacancy_coords, processed_coords)
                except Exception as e:
                    logger.debug(f"Geodesic distance failed in worker: {e}")


        # Возвращаем кортеж: (найденные_координаты, исходный_адрес_кандидата, расстояние)
        return processed_coords, candidate_address, distance

    def __preprocess_cvs(self, df_relevant: pd.DataFrame):
        df_relevant["Описание"] = df_relevant["Описание"].fillna("Нет данных").replace("\n\n", "\n")
        new_cats = {}
        for cat in self.cats_find_cv:
            corpus = [
                "[SEP]".join(
                    [
                        df_relevant.iloc[i]["Описание"] + "\n" + df_relevant.iloc[i]["Опыт работы"]
                            + "\n" + df_relevant.iloc[i]["Категория прав"]
                            + "\n" + df_relevant.iloc[i]["Образование"],
                        ModeInfo.CV,
                        cat,
                        self.category_desc.get(cat, "None"),
                        "None"
                    ]
                )
                for i in range(df_relevant.shape[0])
            ]
            new_cats[cat] = process_corpus(corpus=corpus, func=self.find_info)
            found_values = []
            for i in range(df_relevant.shape[0]):
                info_extracted = new_cats[cat][i]
                info_dict = self.postprocess_extracted_info(info_extracted, cat=cat)
                found_values.append(info_dict[cat])
            df_relevant[cat.capitalize()] = found_values

        return df_relevant

    def rank_second_stage(
        self,
        vacancy: Dict,
        df_relevant: pd.DataFrame,
        df_weights: pd.DataFrame,
        score_threshold_stage_2
    ):
        vacancy_prep = deepcopy(vacancy)

        nan_mask = self.vacancy_mask(vacancy_dict=vacancy_prep)
        logger.info(f"Vacancy NaN mask: {nan_mask}")
        
        vac_desc = vacancy_prep["Описание"]
        
        df_relevant = self.__preprocess_cvs(df_relevant=df_relevant)

        if self.method == Method.EMBEDDINGS:
            logger.info("Computing embeddings")
            for feat in self.feats_match:
                # Проверяем наличие фичи в вакансии и датафрейме
                if feat in vacancy_prep and feat in df_relevant.columns:
                    logger.debug(f"Calculating embedding similarity for '{feat}'...")
                    df_relevant[feat] = df_relevant[feat].fillna("Нет данных")
                    # Оборачиваем в try-except на случай проблем с эмбеддером
                    try:
                        embeddings = self.embedder.generate_embeddings(df_relevant, feat)
                        embeddings_np = np.vstack(embeddings)
                        embedding_vac = self.embedder.embed_corpus([vacancy_prep[feat]])
                        embedding_vac_np = np.array(embedding_vac)
                        cos_sims = cosine_similarity(embedding_vac_np, embeddings_np)[0]
                        df_relevant[f"{feat}_sim"] = cos_sims
                    except Exception as e:
                         logger.error(f"Failed to calculate embedding similarity for '{feat}': {e}. Setting sim to 0.")
                         df_relevant[f"{feat}_sim"] = 0.0
                else:
                     logger.warning(f"Feature '{feat}' for embedding matching not found in vacancy or DataFrame. Skipping.")
                     df_relevant[f"{feat}_sim"] = 0.0 # Ставим 0, если фичи нет

            # Расчет схожести для feats_match_prompt (LLM промпты)
            for feat in self.feats_match_prompt:
                 if feat in vacancy_prep and feat in df_relevant.columns:
                    logger.debug(f"Calculating prompt similarity for '{feat}'...")
                    corpus = []
                    for x in df_relevant[feat].fillna("Нет данных").to_list(): # Тут извлекаем то что отчасти извлечено нейронкой раньше
                        corpus.append("[SEP]".join([f"{feat}:\n" + vacancy_prep[feat], f"{feat}:\n" + "<"+x.replace("\n", " ")+">"]))
                    try:
                        sim_scores = process_corpus(
                            corpus=corpus,
                            func=self.match_prompt,
                            num_workers=self.request_num_workers,
                        )
                        df_relevant[f"{feat}_sim"] = sim_scores
                        logger.debug(f"Prompt similarity for '{feat}' calculated. = {sim_scores}")
                    except Exception as e:
                        logger.error(f"Failed to calculate prompt similarity for '{feat}': {e}. Setting sim to 0.")
                        df_relevant[f"{feat}_sim"] = 0.0
                 else:
                    logger.warning(f"Feature '{feat}' for prompt matching not found in vacancy or DataFrame. Skipping.")
                    df_relevant[f"{feat}_sim"] = 0.0
            logger.info("Finished calculating base similarity scores.")

        active_features = df_weights["Компонента"].tolist()
        sim_scores_names = [f"{feat}_sim" for feat in active_features]

        # Проверяем наличие всех нужных колонок _sim в DataFrame
        missing_cols = [col for col in sim_scores_names if col not in df_relevant.columns]
        if missing_cols:
            logger.warning(f"Missing similarity columns for final score: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                df_relevant[col] = 0.0 # Заполняем нулями, чтобы избежать ошибки

        try:
            # ВВЕРХ перенёс
            #  logger.debug(f"Vacancy for vacancy mask: {vacancy_prep}")
            #  nan_mask = self.vacancy_mask(vacancy_dict=vacancy_prep)
            #  logger.info(f"Vacancy NaN mask: {nan_mask}")
             # Создаем словарь маски для удобного доступа по имени фичи из конфига
             mask_dict = {feat: mask_val for feat, mask_val in zip(self.ranking_features, nan_mask)}
        except Exception as e:
             logger.error(f"Failed to calculate vacancy mask using self.ranking_features: {e}. Using mask of all ones.")
             # Fallback: если ошибка с маской, считаем все поля вакансии заполненными
             mask_dict = {feat: 1 for feat in active_features}

        # Применяем маску к весам из UI в правильном порядке
        try:
            adjusted_weights = np.array([
                df_weights.loc[df_weights['Компонента'] == feat, 'Вес'].fillna(0).iloc[0] * mask_dict.get(feat, 1)
                for feat in active_features # Используем порядок из df_weights
            ])
            logger.info(f"Weights from UI: {df_weights['Вес'].values}")
            logger.info(f"Adjusted weights after mask: {adjusted_weights}")
        except Exception as e:
             logger.error(f"Failed to adjust weights with mask: {e}. Using raw weights.")
             adjusted_weights = df_weights["Вес"].fillna(0).values

        # Считаем итоговый балл
        total_weight = adjusted_weights.sum()
        if total_weight == 0:
            logger.warning("Total adjusted weight for second stage score is zero. Setting score to 0.")
            df_relevant["sim_score_second"] = 0.0
        else:
            try:
                # Убедимся, что используем колонки _sim в том же порядке, что и adjusted_weights
                sim_values_matrix = df_relevant[sim_scores_names].values
                df_relevant["sim_score_second"] = np.dot(sim_values_matrix, adjusted_weights) / total_weight
                logger.info("Calculated final sim_score_second.")
            except Exception as e:
                 logger.error(f"Error calculating final sim_score_second: {e}. Setting score to 0.")
                 df_relevant["sim_score_second"] = 0.0

        # logger.info(f"Final sim_score_second values: {df_relevant['sim_score_second']}")
        if score_threshold_stage_2 > 0.0: # Применяем фильтр, только если порог > 0
            initial_count_before_threshold = len(df_relevant)
            df_relevant = df_relevant[df_relevant["sim_score_second"] >= score_threshold_stage_2].copy()
            logger.info(f"Applied score threshold >= {score_threshold_stage_2:.2f}. "
                        f"Removed {initial_count_before_threshold - len(df_relevant)} candidates. "
                        f"{len(df_relevant)} remaining.")
        else:
            logger.info("Score threshold is 0.0 or not set. Skipping score filtering.")

        df_ranked = df_relevant.sort_values("sim_score_second", ascending=False)
        return df_ranked.head(self.top_n_second_stage), vacancy_prep, nan_mask

if __name__ == "__main__":
    config_path = "./config/config_mass.yaml"
    data_path = "./data_mass/Водитель белаза.csv"
    vac_path = "./data_mass/vacancies.csv"

    ## Load test data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading test data")
    df_relevant = pd.read_csv(data_path)
    vacancy = pd.read_csv(vac_path).iloc[0].to_dict()

    ## Init model
    logger.info("Init model")
    config_model = config["model"]
    selector = MassCvSelector(config=config_model, api_token=os.getenv("OPENAI_TOKEN"))

    ## Make 1st stage ranking
    logger.info("1st stage ranking..")
    df_ranked_1st = selector.rank_first_stage(
        vacancy=vacancy, df_relevant=df_relevant.copy()
    )

    assert df_ranked_1st.shape[0] == config_model["stage_1"]["top_n"], "Wrong size"
    assert (
        df_ranked_1st.iloc[0]["id"] == "№ 2880355971"
    ), "First stage ranking is wrong!"
    logger.info("Finished successfully")

    # logger.info("2nd stage ranking..")
    df_ranked_2nd, vacancy_prep, nan_mask = selector.rank_second_stage(
        vacancy=vacancy, df_relevant=df_ranked_1st.copy()
    )

    assert df_ranked_2nd.shape[0] == config_model["stage_2"]["top_n"], "Wrong size"
    df_ranked_2nd.to_csv("./test_results.csv", index=False)
    assert (
        df_ranked_2nd.iloc[1]["id"] == "№ 2880355971"
    ), "Second stage ranking is wrong!"
    logger.info("Finished successfully")
