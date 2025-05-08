import json
import os
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.base_model import BaseSelector
from utils.enums import Method, ModeInfo
from utils.request_api import OpenAIEmbedder, process_corpus

load_dotenv(override=True)


class ProfSelector(BaseSelector):
    def __init__(
        self, config: Dict, api_token: str, method: Method = Method.EMBEDDINGS
    ):
        super().__init__(config=config, api_token=api_token, method=method)
        self.sim_scores_names = config["stage_2"]["sim_scores_names"]

    def _text_features_intersection(self, str_feats_ref: str, str_feats_match: str):
        set1 = set(str_feats_ref.lower().split(", "))
        set2 = set(str_feats_match.lower().split(", "))
        return len(set1.intersection(set2)) / len(set1)

    def _education_str(self, edu_data_str: str):
        result = ""
        try:
            edu_data = json.loads(edu_data_str)
        except JSONDecodeError:
            return "Нет данных"
        for i, data_item in enumerate(edu_data):
            for key in data_item:
                if data_item[key] is None or data_item[key] == "None":
                    data_item[key] = ""
            data_item_str = f"\n{i + 1}. {data_item['year']} - {data_item['name']}. {data_item['result']}, {data_item['organization']}"
            while not data_item_str[-1].isalpha():
                data_item_str = data_item_str[:-1]
                if len(data_item_str) == 0:
                    break
            if len(data_item_str) == 0:
                continue
            result += data_item_str.replace(" , ", " ") + "."
        if result == "":
            return "Нет данных"
        return result

    def _salary_str(self, salary_data_str: str):
        try:
            salary_data = json.loads(salary_data_str)
        except JSONDecodeError:
            return "Нет данных"
        return f"{salary_data['amount']} {salary_data['currency']}"

    def work_experience_summary(self, json_str: str):
        client = OpenAI(api_key=self.api_token)
        prompt = self.prompt_experience + json_str

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt_experience},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def rank_first_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        ranking_features = self.cluster_match_features.copy()
        for feat in self.text_features:
            df_relevant[f"{feat}_sim"] = (
                df_relevant[feat]
                .fillna("None")
                .apply(lambda x: self.__text_features_intersection(vacancy[feat], x))
            )
            ranking_features.append(f"{feat}_sim")
        self.ranking_features_first_stage = ranking_features

        df_relevant["sim_score_first"] = (
            np.dot(df_relevant[ranking_features].values, self.first_stage_weights)
            / self.first_stage_weights.sum()
        )
        df_ranked = df_relevant.sort_values("sim_score_first", ascending=False)
        return df_ranked.head(self.top_n_first_stage)

    # def __preprocess_vacancy(self, vacancy: Dict):
    #     vacancy["Профессиональная область"] = vacancy.pop("prof_field_full")
    #     vacancy["Должность категория"] = vacancy.pop("Должность_cat")
    #     vacancy["Должность подкатегория"] = vacancy.pop("Должность_subcat")
    #     for cat in self.cats_find_vacancy:
    #         info = "[SEP]".join(
    #             [
    #                 vacancy["Описание"],
    #                 ModeInfo.VACANCY,
    #                 cat,
    #                 self.category_desc.get(cat, "None"),
    #                 vacancy["Список навыков"] if cat == "знания" else "None",
    #             ]
    #         )
    #         info_extracted = self.find_info(info=info)
    #         info_dict = self.postprocess_extracted_info(info=info_extracted, cat=cat)
    #         info_dict[cat.capitalize()] = info_dict.pop(cat)
    #         vacancy.update(info_dict)
    #     vacancy["Full_description"] = self.get_desc(
    #         vacancy=vacancy, keys=self.keys_vacancy
    #     )
    #     return vacancy

    # def preprocess_cvs(self, df_relevant: pd.DataFrame):
    #     corpus_experience = df_relevant["Опыт"].fillna("").to_list()
    #     logger.info("Generating experience summaries")
    #     summaries_experience = process_corpus(
    #         corpus=corpus_experience,
    #         func=self.work_experience_summary,
    #         num_workers=self.request_num_workers,
    #     )
    #     df_relevant = df_relevant.rename(columns={"Опыт": "Опыт raw"})
    #     df_relevant["Опыт"] = summaries_experience
    #     df_relevant["Образование"] = df_relevant["Образование.Высшее"].apply(
    #         lambda x: self.__education_str(x)
    #     )
    #     df_relevant["Зарплатные ожидания"] = df_relevant["Зарплата"].apply(
    #         self.__salary_str
    #     )
    #     df_relevant = df_relevant.rename(
    #         columns={
    #             "prof_field_full": "Профессиональная область",
    #             "Должность_cat": "Должность категория",
    #             "Должность_subcat": "Должность подкатегория",
    #         }
    #     )
    #     df_relevant["Описание"] = df_relevant["Описание"].fillna("Нет данных")
    #     df_relevant["Список навыков"] = df_relevant["Список навыков"].fillna(
    #         "Нет данных"
    #     )
    #     # df_relevant = df_relevant.fillna("Нет данных")
    #     new_cats = {}
    #     for cat in self.cats_find_cv:
    #         corpus = [
    #             "[SEP]".join(
    #                 [
    #                     df_relevant.iloc[i]["Описание"],
    #                     "cv",
    #                     cat,
    #                     self.category_desc.get(cat, "None"),
    #                     df_relevant.iloc[i]["Список навыков"]
    #                     if cat == "знания"
    #                     else "None",
    #                 ]
    #             )
    #             for i in range(df_relevant.shape[0])
    #         ]
    #         new_cats[cat] = process_corpus(corpus=corpus, func=self.find_info)
    #         found_values = []
    #         for i in range(df_relevant.shape[0]):
    #             info_extracted = new_cats[cat][i]
    #             info_dict = self.postprocess_extracted_info(info_extracted, cat=cat)
    #             found_values.append(info_dict[cat])
    #         df_relevant[cat.capitalize()] = found_values

    #     df_relevant["Зарплата"] = df_relevant["Зарплатные ожидания"].copy()
    #     df_relevant["Опыт работы"] = df_relevant["Опыт"].copy()
    #     descs = []
    #     for i in range(len(df_relevant)):
    #         cv_dict = df_relevant.iloc[i].to_dict()
    #         desc = self.__get_desc(cv_dict, keys=self.keys_cv)
    #         descs.append(desc)
    #     df_relevant["Full_description"] = descs
    #     return df_relevant

    def rank_second_stage(self, vacancy: Dict, df_relevant: pd.DataFrame):
        vacancy_prep = self._preprocess_vacancy(vacancy=vacancy)
        vac_desc = vacancy_prep["Full_description"]
        df_relevant = self.preprocess_cvs(df_relevant=df_relevant)

        if self.method == Method.EMBEDDINGS:
            logger.info("Computing descriptions embeddings")
            for feat in self.feats_match:
                df_relevant[feat] = df_relevant[feat].fillna("Нет данных")
                embeddings = self.embedder.generate_embeddings(df_relevant, feat)
                embeddings_np = np.vstack(embeddings)
                embedding_vac = self.embedder.embed_corpus([vacancy_prep[feat]])
                embedding_vac_np = np.array(embedding_vac)
                cos_sims = cosine_similarity(embedding_vac_np, embeddings_np)[0]
                df_relevant[f"{feat}_sim"] = cos_sims
            for feat in self.feats_match_prompt:
                corpus = (
                    df_relevant[feat]
                    .apply(
                        lambda x: "[SEP]".join(
                            [f"{feat}:\n" + vacancy_prep[feat], f"{feat}:\n" + x]
                        )
                    )
                    .to_list()
                )
                sim_scores = process_corpus(
                    corpus=corpus,
                    func=self.match_prompt,
                    num_workers=self.request_num_workers,
                )
                df_relevant[f"{feat}_sim"] = sim_scores
        ## currently not working properly
        elif self.method == Method.PROMPT:
            logger.info("Computing scores with prompt")
            corpus_descs = (
                df_relevant["Full_description"]
                .apply(lambda x: "[SEP]".join([vac_desc, x]))
                .to_list()
            )
            sim_scores = process_corpus(
                corpus=corpus_descs,
                func=self.match_prompt,
                num_workers=self.request_num_workers,
            )
            df_relevant["Full_description_sim"] = sim_scores
        else:
            raise ValueError(
                f"Method doesn't exist: {self.method}, {self.method == Method.PROMPT}, {Method.PROMPT}"
            )

        nan_mask = self.vacancy_mask(vacancy_dict=vacancy_prep)
        weights = self.second_stage_weights * nan_mask
        df_relevant["sim_score_second"] = (
            np.dot(df_relevant[self.sim_scores_names].values, weights) / weights.sum()
        )
        df_ranked = df_relevant.sort_values("sim_score_second", ascending=False)
        df_ranked = df_ranked.rename(
            columns={
                "skills_sim": "Список навыков_sim",
                "prof_field_full_sim": "Профессиональная область_sim",
            }
        )
        return df_ranked.head(self.top_n_second_stage), vacancy_prep, nan_mask


if __name__ == "__main__":
    config_path = "../config/config.yaml"
    data_path = "../../uniweb-demo/notebooks/data_jobs/test_cvs_subset.csv"
    vac_path = "../../uniweb-demo/notebooks/data_jobs/test_vacancy.csv"

    ## Load test data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading test data")
    df_relevant = pd.read_csv(data_path)
    vacancy = pd.read_csv(vac_path, index_col="Unnamed: 0")["45"].to_dict()

    ## Init model
    logger.info("Init model")
    config_model = config["model"]
    selector = CvSelector(config=config_model, api_token=os.getenv("OPENAI_TOKEN"))

    ## Make 1st stage ranking
    logger.info("1st stage ranking..")
    df_ranked_1st = selector.rank_first_stage(
        vacancy=vacancy, df_relevant=df_relevant.copy()
    )

    assert df_ranked_1st.shape[0] == config_model["stage_1"]["top_n"], "Wrong size"
    assert (
        "Некула" in df_ranked_1st.iloc[0]["Описание"]
    ), "First stage ranking is wrong!"
    logger.info("Finished successfully")

    logger.info("2nd stage ranking..")
    df_ranked_2nd, vacancy_prep, nan_mask = selector.rank_second_stage(
        vacancy=vacancy, df_relevant=df_ranked_1st.copy()
    )

    assert df_ranked_2nd.shape[0] == config_model["stage_2"]["top_n"], "Wrong size"
    df_ranked_2nd.to_csv("./test_results.csv", index=False)
    assert (
        "80000 RUR" in df_ranked_2nd.iloc[0]["Зарплата"]
    ), "Second stage ranking is wrong!"
    logger.info("Finished successfully")
