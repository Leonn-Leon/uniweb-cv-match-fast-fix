import os
from collections import OrderedDict
from enum import StrEnum

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.models import MassSelector, ProfSelector

load_dotenv(override=True)


def df2dict(df: pd.DataFrame):
    res = OrderedDict()
    for i in range(df.shape[0]):
        row = df.iloc[i]
        res[row["ID"]] = row.to_dict()
    return res


def select_color(match_score: int):
    if match_score == -1:
        return "grey"
    if match_score >= 70:
        return "green"
    elif 40 < match_score < 70:
        return "orange"
    else:
        return "red"


# @st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


# @st.cache_resource
def load_model(config_path: st):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config_model = config["model"]
    if "mass" in config_path:
        selector = MassSelector(
            config=config_model, api_token=os.getenv("OPENAI_TOKEN")
        )
    else:
        selector = ProfSelector(config=config_model, api_token=os.getenv("OPENAI_TOKEN"))
    return selector, config


def format_intersection(str_vac: str, str_cv) -> str:
    str_vac = str_vac.replace("\n\n", ", ").replace("\n", ", ")
    str_cv = str_cv.replace("\n\n", ", ").replace("\n", ", ")
    split_cv = str_cv.lower().split(", ")
    set1 = set(str_vac.lower().split(", "))
    set2 = set(split_cv)
    intersection = set1.intersection(set2)
    for i, word in enumerate(split_cv):
        if word in intersection:
            split_cv[i] = f"**{word}**"
    return ", ".join(split_cv).replace(":**,", ":**").replace(":,", ":")


class Mode(StrEnum):
    MASS = "Массовый"
    PROF = "Профессиональный"
