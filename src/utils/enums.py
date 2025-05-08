from enum import StrEnum


class Method(StrEnum):
    EMBEDDINGS = "Метод 1"
    PROMPT = "Метод 2"


class ModeInfo(StrEnum):
    VACANCY = "vacancy"
    CV = "cv"
