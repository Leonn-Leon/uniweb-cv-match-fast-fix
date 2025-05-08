from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from typing import List, Union, Dict
from rank_bm25 import BM25Okapi
import json
from json import JSONDecodeError
import simplemma
import re
from ast import literal_eval
from geopy.distance import geodesic
import simplemma
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import requests
import re
import os

from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from utils.enums import Method, ModeInfo
from utils.request_api import OpenAIEmbedder, process_corpus
from dotenv import load_dotenv
import nltk

class BaseSelector(ABC):
    def __init__(self, config: Dict, api_token: str, method: Method = Method.EMBEDDINGS):
        """Инициализация базовой модели"""
        
        load_dotenv(override=True)
        nltk.download("punkt_tab")

        self.config = config
        ## stage 1 config
        self.text_features = config["stage_1"]["text_features"]
        self.cluster_match_features = config["stage_1"]["cluster_match_features"]
        self.first_stage_weights = np.array(config["stage_1"]["weights"])
        self.top_n_first_stage = config["stage_1"]["top_n"]
        self.ranking_features_first_stage = None
        ## stage 2 config
        self.keys_vacancy = config["stage_2"]["keys_vacancy"]
        self.model_name = config["stage_2"]["model_name"]

        self.prompt_experience = config["stage_2"]["prompt_experience"]
        # self.system_prompt_experience = config["stage_2"]["system_prompt_experience"]

        self.prompt_info = config["stage_2"]["prompt_info"]
        self.system_prompt_info = config["stage_2"]["system_prompt_info"]
        self.question_vac = config["stage_2"]["question_vac"]
        self.question_cv = config["stage_2"]["question_cv"]
        self.category_desc = config["stage_2"]["category_desc"]
        self.cats_find_vacancy = config["stage_2"]["cats_find_vacancy"]
        self.cats_find_cv = config["stage_2"]["cats_find_cv"]

        self.request_num_workers = config["stage_2"]["request_num_workers"]
        self.keys_cv = config["stage_2"]["keys_cv"]
        self.feats_match = config["stage_2"]["feats_match"]
        self.feats_match_prompt = config["stage_2"]["feats_match_prompt"]
        self.ranking_features = config["stage_2"]["ranking_features"]
        self.second_stage_weights = np.array(config["stage_2"]["weights"])
        self.top_n_second_stage = config["stage_2"]["top_n"]

        self.embedder = OpenAIEmbedder(
            api_key=api_token, model_name=config["stage_2"]["model_name_embed"]
        )
        self.api_token = api_token
        self.method = method
        if method not in [Method.EMBEDDINGS, Method.PROMPT]:
            self.method = str(Method.EMBEDDINGS)
        self.prompt_matching = config["stage_2"]["prompt_matching"]
        self.system_prompt_matching = config["stage_2"]["system_prompt_matching"]
        self.stemmer = SnowballStemmer(language="russian")
        
    @abstractmethod
    def rank_first_stage(self, vacancy, df_relevant, *args, **kwargs):
        """Абстрактный метод первого этапа ранжирования"""
        pass

    @abstractmethod
    def rank_second_stage(self, vacancy, df_relevant, *args, **kwargs):
        """Абстрактный метод второго этапа ранжирования"""
        pass

    @abstractmethod
    def preprocess_vacancy(self, vacancy):
        """Базовая предобработка вакансии (может быть переопределена в наследниках)"""
        pass

    @abstractmethod
    def preprocess_cvs(self, df_relevant: pd.DataFrame):
        pass

    def vacancy_mask(self, vacancy_dict: Dict):
        mask_vac = [
            True
            if feat not in vacancy_dict
            else int(
                vacancy_dict[feat].lower().strip()
                not in [
                    "нет данных",
                    "нет информации",
                    "",
                    "none",
                    "не указано",
                    "не указана",
                    "не указан",
                    "не задан",
                ]
            )
            for feat in self.ranking_features
        ]
        return np.array(mask_vac)

    def postprocess_extracted_info(self, info: str, cat: str):
        try:
            info_dict = json.loads(info)
            if type(info_dict[cat]) == list:
                info_dict[cat] = ", ".join(info_dict[cat])
            if type(info_dict[cat]) == dict:
                info_dict[cat] = "; ".join(
                    [f"{key}: {value}" for key, value in info_dict[cat].items()]
                )
            if (
                (info_dict[cat] is None)
                or (info_dict[cat] == "None")
                or (info_dict[cat] == "")
            ):
                info_dict[cat] = "Нет данных"
        except Exception:
            info_dict = {cat: "Нет данных"}
        return info_dict

    def match_prompt(self, data: str):
        client = OpenAI(api_key=self.api_token)
        vac_desc, cv_desc = data.split("[SEP]")
        prompt = self.prompt_matching + f"\n{cv_desc}"
        system_prompt = self.system_prompt_matching + f"\n{vac_desc}"
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        result = completion.choices[0].message.content
        try:
            score = json.loads(result)["match_score"]
        except JSONDecodeError:
            score = 0.5
        return score

    def get_desc(self, vacancy: Dict, keys: List[str]):
        description_items = []
        for key in keys:
            if key in vacancy:
                description_items.append(f"{key}:\n{vacancy[key]}")
        return "\n\n".join(description_items)

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
            response = requests.get(base_url, params=params, timeout=10)
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

    def _shorten_address(self, address: str):
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

    def _score_move(self, move: str, dist: float, dist_thresh: float = 50):
        if dist < dist_thresh:
            return 1.0
        value_array = np.array(["Невозможен", "Нет данных", "Возможен"])
        return float(np.where(value_array == move)[0][0]) / 2

    def _normalize_text(self, text: str):
        text = text.lower().replace("\n", " ")
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = " ".join(sorted(text.split(" ")))
        return text

    def _minmax_scale(self, x: np.ndarray):
        range_ = x.max() - x.min()
        if range_ == 0:
            return np.zeros_like(x)
        return (x - x.min()) / range_

    def _tokenize_feat(self, text: str):
        norm_text = self._normalize_text(text)
        words = word_tokenize(norm_text, language="russian")
        prep_words = []
        for word in words:
            word_lemm = simplemma.lemmatize(word, lang="ru")
            word_stemm = self.stemmer.stem(word_lemm)
            prep_words.append(word_stemm)
        return prep_words

    def _bm25_score(self, feature_list: List[str], query: str):
        tokenized_corpus = [self._tokenize_feat(x) for x in feature_list]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self._tokenize_feat(query)
        scores = np.abs(bm25.get_scores(tokenized_query))
        return self._minmax_scale(scores)

    def find_geo_distance(
        self, coords_vac: Union[str, tuple], coords_cand: Union[str, tuple]
    ):
        if type(coords_vac) == str:
            coords_vac = literal_eval(coords_vac)
        if type(coords_cand) == str:
            coords_cand = literal_eval(coords_cand)
        return geodesic(coords_vac, coords_cand).kilometers

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

    def _normalize_weights(self, weights):
        """Нормализация весов"""
        weights = np.array(weights)
        total = weights.sum()
        if total == 0:
            return weights
        return weights / total
