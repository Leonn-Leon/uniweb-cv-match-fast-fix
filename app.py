# app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from copy import deepcopy
from loguru import logger
import datetime 
import requests
import re
import time
from urllib.parse import quote
from src.utils.scraper_api import build_search_params, launch_pipeline, track_task_progress
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Импорт ваших модулей
from src.utils.utils import Mode, load_data, load_model, df2dict
from src.ui import common_ui, mass_ui, prof_ui # Предполагается, что эти модули существуют

# --- Инициализация Session State ---
if "computed" not in st.session_state:
    st.session_state["computed"] = False
if "vahta_mode" not in st.session_state:
    st.session_state["vahta_mode"] = False
if "distance_option" not in st.session_state:
    st.session_state["distance_option"] = "Нет ограничений"
if "original_address_weight" not in st.session_state: 
    st.session_state.original_address_weight = None

# Новые ключи для session_state
if "huntflow_vacancies_list" not in st.session_state: # Список для selectbox (отображаемое_имя, id)
    st.session_state.huntflow_vacancies_list = [] 
if "huntflow_vacancies_details" not in st.session_state: # Словарь {vacancy_id: полные_детали_вакансии}
    st.session_state.huntflow_vacancies_details = {} 
if "selected_huntflow_vacancy_id" not in st.session_state: # ID выбранной вакансии из Huntflow
    st.session_state.selected_huntflow_vacancy_id = None

# Ключи для предзаполнения полей формы (предполагается, что mass_ui.py их использует)
if "vacancy_form_title" not in st.session_state:
    st.session_state.vacancy_form_title = ""
if "vacancy_form_required" not in st.session_state:
    st.session_state.vacancy_form_required = ""
if "vacancy_form_location" not in st.session_state:
    st.session_state.vacancy_form_location = ""
if "vacancy_form_optional" not in st.session_state:
    st.session_state.vacancy_form_optional = ""
# --- Конец инициализации Session State ---

st.title("Подбор кандидатов 💼")

current_mode = Mode.MASS # Пока жестко задаем массовый подбор

# --- Загрузка данных и модели в зависимости от режима ---
if current_mode == Mode.PROF:
    selector, config = load_model(config_path="./config/config.yaml")
else: # Mode.MASS
    selector, config = load_model(config_path="./config/config_mass.yaml")


with open("data/processed/mass/resual_regions.json", "r", encoding="utf-8") as f:
    resual_regions = json.load(f)

if not selector or not config:
    st.error("Не удалось загрузить модель или конфигурацию. Работа приложения остановлена.")
    st.stop()

if 'current_threshold' not in st.session_state: # Порог по умолчанию
    st.session_state['current_threshold'] = config["model"]["stage_2"]["score_threshold"]


# --- Callback для чекбокса "Вахта" ---
def update_address_weight_callback():
    try:
        df = st.session_state["df_weights"]
        is_vahta = st.session_state["vahta_mode"]
        address_index_list = df.index[df['Компонента'] == 'Адрес'].tolist()
        if not address_index_list: return
        address_index = address_index_list[0]
        current_weight = df.loc[address_index, 'Вес']
        if is_vahta:
            if current_weight != 0.0:
                st.session_state.original_address_weight = current_weight
                df.loc[address_index, 'Вес'] = 0.0
                st.toast("Вес 'Адрес' установлен в 0 (Вахта).", icon="⚠️")
        else:
            if st.session_state.original_address_weight is not None:
                if current_weight == 0.0: 
                    df.loc[address_index, 'Вес'] = st.session_state.original_address_weight
                    st.toast(f"Вес 'Адрес' восстановлен: {st.session_state.original_address_weight}", icon="👍")
        st.session_state["df_weights"] = df
    except Exception as e:
        st.error(f"Ошибка при обновлении веса 'Адрес': {e}")

HUNTFLOW_BASE_URL = st.secrets.get("HUNTFLOW_BASE_URL")

def get_huntflow_proxies():
    proxy_user = st.secrets.get("HUNTFLOW_PROXY_USER")
    proxy_pass = st.secrets.get("HUNTFLOW_PROXY_PASS")
    proxy_host = st.secrets.get("HUNTFLOW_PROXY_HOST")
    proxy_port = st.secrets.get("HUNTFLOW_PROXY_PORT")

    if all([proxy_user, proxy_pass, proxy_host, proxy_port]):
        proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        return {
            "http": proxy_url,
            "https": proxy_url,
        }
    return None

import json
import requests
import streamlit as st
from pathlib import Path

def fetch_huntflow_vacancies(page_size: int = 100,
                                       save_dir: str = "vacancies_json"):
    """
    Скачивает все вакансии постранично, записывает каждый объект в отдельный JSON-файл,
    а в session_state кладёт список кортежей (название, id).
    """
    # Проверяем token
    hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
    hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
    if not hf_token:
        st.error("HUNTFLOW_API_TOKEN не задан в secrets.toml")
        return [], {}

    headers = {"Authorization": f"Bearer {hf_token}"}
    url = f"{HUNTFLOW_BASE_URL}/accounts/{hf_account_id}/vacancies"
    proxies = get_huntflow_proxies()

    # Папка для хранения JSON-файлов
    out_folder = Path(save_dir)
    out_folder.mkdir(exist_ok=True)

    vacancy_list = []
    vacancy_details = {}

    offset = 0
    while True:
        params = {
            "state": "OPEN",
            "count": page_size,
            "page": int(offset/page_size)+1,
            "opened": False
        }
        try:
            resp = requests.get(url, headers=headers, params=params, proxies=proxies)
            resp.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Ошибка при запросе Huntflow API: {e}, response: {resp.json()}")
            break

        logger.debug(f"Загружен лист вакансий")

        items = resp.json().get("items", [])
        if not items:
            break

        for vac in items:
            vac_id = vac.get("id")
            title = vac.get("position", f"Вакансия #{vac_id}")
            # Сохраняем метаданные списка
            vacancy_list.append((title, vac_id))
            # Сохраняем полный JSON-объект в файл
            file_path = out_folder / f"vacancy_{vac_id}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(vac, f, ensure_ascii=False, indent=2)

            # При необходимости оставить в памяти лишь минимальный маппинг
            vacancy_details[vac_id] = str(file_path)

        offset += len(items)
        # Можно показывать прогресс
        # st.write(f"Загружено и сохранено: {offset} вакансий…")
        # logger.info(items[0])

    return vacancy_list, vacancy_details

def get_hh_contacts_api(resume_id, access_token_ext=None):
    """Получает контакты кандидата с HH.ru."""
    try:
        hh_token = access_token_ext
        if not hh_token: st.error("Токен HH.ru API не найден."); return None
        headers = {"Authorization": f"Bearer {hh_token}", "User-Agent": "Uniweb CV Match App"}
        resume_url = f"https://api.hh.ru/resumes/{resume_id}?get_with_contact=true"
        r_resume = requests.get(resume_url, headers=headers)
        logger.debug(f"Получены данные:"+ str(r_resume.json()))
        r_resume.raise_for_status()
        r_contacts = r_resume.json().get("contact")
        print(f"Контакты: {r_contacts}", type(r_contacts))
        if not r_contacts or (isinstance(r_contacts, list) and len(r_contacts) == 0):
            logger.debug("Делаем запрос контактов глубже")
            contacts_href = r_resume.json().get("actions").get('get_with_contact').get("url")
            if not contacts_href:
                st.warning(f"Нет ссылки на контакты HH {resume_id}."); return None
            r_resume = requests.get(contacts_href, headers=headers)
            logger.debug(f"Глубокие контакты: {r_resume.json()}")
            r_resume.raise_for_status()
            r_contacts = r_resume.json().get("contact")
        details = {"first_name": r_resume.json().get("first_name", ""), "last_name": r_resume.json().get("last_name", ""),
                   "middle_name": r_resume.json().get("middle_name", ""), "phone": "", "email": "empty_email@error.ru"}
        # if r_contacts[0].get("type").get("id") == "email":
        #     details["email"] = r_contacts[0].get("value", "")
        #     details["phone"] = r_contacts[1].get("value", "").get("formatted")
        # elif r_contacts[0].get("type").get("id") == "cell":
        #     details["phone"] = r_contacts[0].get("value").get("formatted")
        #     details["email"] = r_contacts[1].get("value")
        for contact in r_contacts:
            if contact.get("type").get("id") == "email":
                details["email"] = contact.get("value", "")
            elif contact.get("type").get("id") == "cell":
                details["phone"] = contact.get("value").get("formatted")
        return details
    except requests.exceptions.RequestException as e: st.error(f"Ошибка API HH.ru ({resume_id}): {e}"); return {}
    except Exception as e: st.error(f"Ошибка (HH contacts {resume_id}): {e}"); return {}

def get_avito_contacts_api(resume_id, access_token_ext=None):
    """Получает контакты кандидата с Avito."""
    try:
        avito_token = access_token_ext
        if not avito_token: st.error("Токен Avito API не найден."); return None
        headers = {"Authorization": f"Bearer {avito_token}", "Accept": "application/json", "User-Agent": "Uniweb CV Match App"}
        url = f"https://api.avito.ru/job/v1/resumes/{resume_id}/contacts/"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        logger.debug(response.json())
        FIO = response.json().get("full_name")
        if FIO:
            details = {"first_name": FIO.get("first_name", ""), "last_name": FIO.get("last_name", ""),
                    "middle_name": FIO.get("patronymic", ""), "phone": "", "email": "empty_email@error.ru"}
        else:
            details = {"first_name": response.json().get("name", ""), "last_name": "",
                       "middle_name": "", "phone": "", "email": "empty_email@error.ru"}
        r_contacts = response.json().get("contacts")
        for contact in r_contacts:
            if contact.get("type") == "e-mail":
                details["email"] = contact.get("value", "")
            elif contact.get("type") == "phone":
                details["phone"] = contact.get("value", "")
        return details
    except requests.exceptions.RequestException as e: st.error(f"Ошибка API Avito ({resume_id}): {e}"); return None
    except Exception as e: st.error(f"Ошибка (Avito contacts {resume_id}): {e}"); return None

def _extract_pii_details(pii_data, source):
    """Извлекает стандартизированные PII из ответа HH/Avito."""
    details = {"first_name": "Неизв.", "last_name": "Неизв.", "middle_name": "", "phone": "", "email": ""}
    if not pii_data: return details

    details["first_name"] = pii_data.get("first_name", "Неизв.")
    details["last_name"] = pii_data.get("last_name", "Неизв.")
    details["middle_name"] = pii_data.get("middle_name", "")
    details["phone"] = pii_data.get("phone", "")
    details["email"] = pii_data.get("email", "")
    return details


def create_huntflow_applicant_api(pii_standardized, candidate_ml_data, source_resume_id, source_type_for_hf):
    """Создает кандидата в Huntflow (шаг 1)."""
    proxies = get_huntflow_proxies()
    try:
        hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
        hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        if not hf_token: return None, "Токен Huntflow API не найден."
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        url = f"{HUNTFLOW_BASE_URL}/accounts/{hf_account_id}/applicants"
        body = {
            "last_name": pii_standardized["last_name"], "first_name": pii_standardized["first_name"],
            "middle_name": pii_standardized["middle_name"], "phone": pii_standardized["phone"],
            "email": pii_standardized["email"],
            "position": candidate_ml_data.get("Должность", ""),
            "money": str(candidate_ml_data.get("Зарплата", "")),
            "photo": None,
            "externals": [{"auth_type": "NATIVE", "id": source_resume_id, 
                           "data": {"body": f"Кандидат из {source_type_for_hf}. Скоринг: {candidate_ml_data.get('sim_score_second', '')}%"} }]
        }
        logger.debug(f"Создание кандидата в HF: {body}")
        response = requests.post(url, headers=headers, json=body, proxies=proxies, timeout=10)
        logger.debug(f"Создан кандидат в HF: {response.json()}")
        response.raise_for_status()
        return response.json().get("id"), None
        # return source_resume_id, None
    except requests.exceptions.RequestException as e: return None, f"API HF (создание): {e.response.text if e.response else e}"
    except Exception as e: return None, f"Ошибка HF (создание): {e}"

def fill_huntflow_questionary_api(hf_applicant_id, candidate_ml_data):
    """Заполняет анкету кандидата в Huntflow (шаг 2, упрощенно)."""
    try:
        proxies = get_huntflow_proxies()

        hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
        hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        if not hf_token: return False, "Токен Huntflow API не найден."
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        url = f"{HUNTFLOW_BASE_URL}/accounts/{hf_account_id}/applicants/{hf_applicant_id}/questionary"
        body = { # Упрощено - нужны ID из справочников для многих полей
            "experience": int(candidate_ml_data.get("Опыт работы (лет)", 0)) if candidate_ml_data.get("Опыт работы (лет)") else None,
            "experience_description": candidate_ml_data.get("Опыт работы", "") # Текстовое описание
        }
        body_filtered = {k: v for k, v in body.items() if v is not None}
        if not body_filtered: return True, "Нет данных для анкеты."
        response = requests.post(url, headers=headers, json=body_filtered, proxies=proxies, timeout=10)
        response.raise_for_status()
        return True, None
    except requests.exceptions.RequestException as e: return False, f"API HF (анкета): {e.response.text if e.response else e}"
    except Exception as e: return False, f"Ошибка HF (анкета): {e}"

def link_applicant_to_vacancy_api(hf_applicant_id, hf_vacancy_id, score_percentage):
    """Привязывает кандидата к вакансии в Huntflow."""
    try:
        proxies = get_huntflow_proxies()
        hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
        hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        if not hf_token: return False, "Токен Huntflow API не найден."
        if not hf_vacancy_id: return True, "ID вакансии HF не указан."
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        url = f"{HUNTFLOW_BASE_URL}/accounts/{hf_account_id}/applicants/{hf_applicant_id}/vacancy"
        body = {
            "vacancy": hf_vacancy_id, "status": 21, # Статус из ТЗ
            "comment": f"Автоматически добавлен после скоринга. Результат: {score_percentage}%"
        }
        response = requests.post(url, headers=headers, json=body, proxies=proxies, timeout=10)
        response.raise_for_status()
        return True, None
    except requests.exceptions.RequestException as e: return False, f"API HF (привязка): {e.response.text if e.response else e}"
    except Exception as e: return False, f"Ошибка HF (привязка): {e}"

def parse_resume_link(link_url):
    """
    Извлекает источник ('hh' или 'avito') и ID резюме из URL.
    Возвращает кортеж (source, resume_id) или (None, None) в случае неудачи.
    """
    if not link_url or not isinstance(link_url, str):
        logger.warning(f"Получена некорректная ссылка: {link_url}")
        return None, None

    hh_match = re.search(r"hh\.ru/resume/([a-f0-9]{32,})", link_url, re.IGNORECASE) # HH ID обычно длинный hex
    if hh_match:
        return "hh", hh_match.group(1)
    
    avito_match = re.search(r"avito\.ru/job/v1/resumes/(\d+)/contacts/?", link_url, re.IGNORECASE)
    if avito_match:
        return "avito", avito_match.group(1)

    # Дополнительные проверки, если основные паттерны не сработали (менее точные)
    if "hh.ru" in link_url:
        # Попытка извлечь ID из более общих ссылок HH, если они есть в ваших данных
        hh_generic_match = re.search(r"hh\.ru/.*\?resume=([a-f0-9]+)", link_url, re.IGNORECASE)
        if hh_generic_match:
            return "hh", hh_generic_match.group(1)
        logger.warning(f"Ссылка HH.ru не соответствует известным паттернам для извлечения ID: {link_url}")
        return "hh", None # Источник известен, ID не извлечен
        
    if "avito.ru" in link_url:
         # Попытка извлечь ID из более общих ссылок Avito
        avito_generic_match = re.search(r"avito\.ru/(?:.+/)?(?:rezume(?:/.+)?_|\w+/|items/)(\d+)", link_url, re.IGNORECASE)
        if avito_generic_match:
            return "avito", avito_generic_match.group(1)
        logger.warning(f"Ссылка Avito не соответствует известным паттернам для извлечения ID: {link_url}")
        return "avito", None # Источник известен, ID не извлечен

    logger.warning(f"Не удалось определить источник или ID из ссылки: {link_url}")
    return None, None

def get_hh_oauth_token():
    refresh_token = st.secrets.get("HH_REFRESH_TOKEN")

    logger.info("HH: Попытка получения нового access token...")
    token_url = st.secrets.get("HH_TOKEN_URL")

    payload = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(token_url, data=payload, headers=headers)
        logger.info("Получили при запросе на обновление токена: "+str(response.json()))
        response.raise_for_status() # Проверка на HTTP ошибки
        token_data = response.json()
        
        new_access_token = token_data.get('access_token')

        if new_access_token:
            st.secrets["HH_API_TOKEN"] = new_access_token
            st.secrets["HH_REFRESH_TOKEN"] = token_data.get('refresh_token')
            st.success("HH.ru: Access token успешно получен/обновлен.")
            logger.info("HH.ru: Access token успешно получен/обновлен.")
            return new_access_token
        else:
            # st.error(f"HH.ru: Не удалось получить 'access_token' из ответа. Ответ: {token_data}")
            logger.error(f"HH.ru: Не удалось получить 'access_token' из ответа. Ответ: {token_data}")
            return st.secrets["HH_API_TOKEN"]
    except requests.exceptions.RequestException as e:
        # st.error(f"HH.ru: Ошибка при получении токена: {e}")
        logger.error(f"HH.ru: Ошибка при получении токена: {e}")
        return st.secrets["HH_API_TOKEN"]

AVITO_ACCESS_TOKEN_KEY = "avito_access_token_val"
AVITO_TOKEN_EXPIRES_AT_KEY = "avito_token_expires_at_val"

def get_avito_oauth_token():
    access_token = st.session_state.get(AVITO_ACCESS_TOKEN_KEY)
    expires_at = st.session_state.get(AVITO_TOKEN_EXPIRES_AT_KEY)

    if access_token and expires_at and time.time() < expires_at:
        logger.debug("Avito: Используется существующий валидный токен.")
        return access_token

    logger.info("Avito: Попытка получения нового access token...")
    token_url = st.secrets.get("AVITO_TOKEN_URL") # Должен быть "https://api.avito.ru/token/"
    client_id = st.secrets.get("AVITO_CLIENT_ID")
    client_secret = st.secrets.get("AVITO_CLIENT_SECRET")

    if not all([token_url, client_id, client_secret]):
        st.error("Avito: CLIENT_ID, CLIENT_SECRET или TOKEN_URL не настроены в секретах.")
        logger.error("Avito: CLIENT_ID, CLIENT_SECRET или TOKEN_URL не настроены в секретах.")
        return None

    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(token_url, data=payload, headers=headers)
        response.raise_for_status()
        token_data = response.json()

        new_access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 86400) # Время жизни в секундах, по дефолту сутки (86400 сек)

        if new_access_token and isinstance(expires_in, int):
            st.session_state[AVITO_ACCESS_TOKEN_KEY] = new_access_token
            st.session_state[AVITO_TOKEN_EXPIRES_AT_KEY] = time.time() + expires_in - 60 # Буфер
            st.success("Avito: Access token успешно получен/обновлен.")
            logger.info("Avito: Access token успешно получен/обновлен.")
            return new_access_token
        else:
            st.error(f"Avito: Не удалось получить 'access_token' или 'expires_in' из ответа. Ответ: {token_data}")
            logger.error(f"Avito: Не удалось получить 'access_token' или 'expires_in' из ответа. Ответ: {token_data}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Avito: Ошибка при получении токена: {e}")
        logger.error(f"Avito: Ошибка при получении токена: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                st.json(e.response.json())
                logger.error(f"Avito: Тело ошибки: {e.response.json()}")
            except ValueError:
                st.text(e.response.text)
                logger.error(f"Avito: Тело ошибки (не JSON): {e.response.text}")
        return None
    
def get_region_name_by_id(region_rusal_id):
    # region_rusal_id должен быть int
    for region in resual_regions.get("fields", []):
        if region.get("id") == region_rusal_id:
            return region.get("name")
    return ""

# CHAT2DESK_BASE_URL = "https://m.bot-marketing.com/api/public/tunnelSessions"
CHAT2DESK_BASE_URL = "https://ror.chat2desk.com/webhooks/smart_script/LoC5G7r"

def sanitize_vacancy_name(name_str):
    """Удаляет специальные символы и оставляет только буквы, цифры, пробелы, тире, точки."""
    if not name_str or not isinstance(name_str, str):
        return "Не указано"
    # Удаляем все, кроме букв (русских и английских), цифр, пробелов, тире, точек
    # Также заменяем множественные пробелы на один
    sanitized = re.sub(r'[^\w\s\.\-а-яА-ЯёЁ]', '', name_str)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized if sanitized else "Не указано"


def send_to_chat2desk_api(phone_number: str, 
                          client_name: str, 
                          hf_account_id: str, 
                          hf_vacancy_id: str, 
                          hf_applicant_id: str, 
                          vacancy_name_original: str):
    """Отправляет запрос на инициацию диалога в Chat2Desk."""
    
    chat2desk_session_id = st.secrets.get("CHAT2DESK_SESSION_ID", "bphgdRze7")
    chat2desk_code = st.secrets.get("CHAT2DESK_CODE", "rusal")

    if not chat2desk_session_id or not chat2desk_code:
        msg = "Chat2Desk: CHAT2DESK_SESSION_ID или CHAT2DESK_CODE не найдены в секретах."
        st.error(msg)
        logger.error(msg)
        return False, "Ошибка конфигурации Chat2Desk"

    # Предполагаем, что phone_number уже в формате 7XXXXXXXXXX
    phone_number = re.sub(r"\D", "", phone_number)
    if not (phone_number and phone_number.startswith('7') and len(phone_number) == 11 and phone_number.isdigit()):
        msg = f"Некорректный формат номера телефона для Chat2Desk: {phone_number}. Ожидается 7XXXXXXXXXX."
        st.error(msg)
        logger.error(f"Chat2Desk: {msg}")
        return False, msg
    
    sanitized_vacancy_name_str = sanitize_vacancy_name(vacancy_name_original)

    params_for_url = {
        "code": chat2desk_code,
        "params[phonenumber]": phone_number,
        "params[nameclient]": client_name,
        "params[account_id]": str(hf_account_id),
        "params[vacancy_id]": str(hf_vacancy_id) if hf_vacancy_id else "",
        "params[application_id]": str(hf_applicant_id),
        "params[vacancyname]": sanitized_vacancy_name_str
    }

    param_pairs = [f"{quote(key)}={quote(str(value))}" for key, value in params_for_url.items()]
    url_with_params = f"{CHAT2DESK_BASE_URL}?{'&'.join(param_pairs)}"

    logger.info(f"Chat2Desk: Запрос на URL: {url_with_params}")
    
    ##############

    # params_for_url = {
    #     "code": chat2desk_code,
    #     "params[phonenumber]": 77770444258,
    #     "params[nameclient]": "Вячеслав",
    #     "params[account_id]": str(2),
    #     "params[vacancy_id]": str(24295) if hf_vacancy_id else "",
    #     "params[application_id]": str(469314),
    #     "params[vacancyname]": "Эксперт"
    # }
    
    # param_pairs = [f"{quote(key)}={quote(str(value))}" for key, value in params_for_url.items()]
    # url_with_params = f"{CHAT2DESK_BASE_URL}?{'&'.join(param_pairs)}"

    # logger.info(f"Chat2Desk: Запрос на URL TEST: {url_with_params}")

    ############33

    try:
        response = requests.post(url_with_params, headers={'Content-Type': 'application/json'}, timeout=5) 
        response.raise_for_status()
        
        success_message = f"Запрос в Chat2Desk отправлен (Статус: {response.status_code}). Ответ: {response.text[:150]}"
        logger.info(f"Chat2Desk: Успешный ответ: {success_message}")
        return True, success_message

    except requests.exceptions.HTTPError as e:
        error_msg = f"Chat2Desk HTTP ошибка: {e.response.status_code} - {e.response.text}"
    except requests.exceptions.RequestException as e:
        error_msg = f"Chat2Desk ошибка сети: {e}"
    except Exception as e:
        error_msg = f"Chat2Desk неожиданная ошибка: {e}"
    
    st.error(error_msg)
    logger.error(error_msg)
    return False, error_msg

RENAME_MAP = {
    "resume_schema.link":                       "link",
    "resume_schema.address":                    "address",
    "resume_schema.mainParams.Тип занятости":   "Тип занятости",
    "resume_schema.mainParams.График работы":   "График работы",
    "resume_schema.additionalParams.Образование": "Образование",
    "resume_schema.additionalParams.Опыт работы": "Опыт работы",
    "resume_schema.additionalParams.Категория прав": "Категория прав",
    "resume_schema.description":                "Описание",
    "resume_schema.title":                      "Должность",
    "id":                                       "ID",
}

# какие колонки гарантированно списковые — склеиваем через '\n'
LIST_LIKE_COLS: List[str] = [
    "Образование",
    "Опыт работы",
    "Компетенции",
]

def _scrape_single_aggregator(vacancy_details: dict, aggregator: str) -> pd.DataFrame | None:
    """Возвращает extra_df для одного агрегатора либо None."""
    logger.info(f"Собираем параметры для скрапинга.")
    search_params = build_search_params(vacancy_details, aggregator)
    if not search_params:
        return None

    logger.info(f"Запускаем скрапинг {aggregator}, search_params {search_params}")
    task_id = launch_pipeline(search_params, aggregator)
    if not task_id:
        return None

    logger.info(f"Ждём завершения работы скрапинга для {aggregator}")
    task_info = track_task_progress(task_id)
    if not task_info or task_info.get("status") != "success":
        logger.info(f"Таска провалилась для {aggregator}, результат: {task_info}")
        return None

    candidates = task_info.get("result", {}).get("candidates", [])
    if not candidates:
        logger.error(f"Нет результатов для {aggregator}, {task_info["result"]["candidates"]}")
        logger.error(str(task_info)[:1000])
        return None
    
    logger.success(f"Есть результаты для {aggregator}")
    
    df = pd.json_normalize(candidates, sep='.')
    return df


def append_extra_resumes(df_cv: pd.DataFrame, vacancy_details: dict) -> pd.DataFrame:
    """
    Одновременный поиск резюме на HH и Avito, склейка в исходный df_cv.
    """
    extra_frames: List[pd.DataFrame] = []
    aggregators = ("headhunter_api", "avito_api")

    logger.info("Начинаем скрапинг hh и avito")

    # --- 1. параллельный запуск двух пайплайнов --------------------------------
    with ThreadPoolExecutor(max_workers=len(aggregators)) as pool:
        futures = {
            pool.submit(_scrape_single_aggregator, vacancy_details, agg): agg
            for agg in aggregators
        }
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None:
                extra_frames.append(df)

    # --- 2. если что-то нашли, готовим к склейке --------------------------------
    if not extra_frames:
        st.info("Дополнительных резюме не найдено.")
        return df_cv

    extra_all = pd.concat(extra_frames, ignore_index=True)

    # 2.1. применяем карту переименования
    cols_to_rename = {old: new for old, new in RENAME_MAP.items() if old in extra_all.columns}
    extra_all = extra_all.rename(columns=cols_to_rename)

    # 2.2. списки → строка
    for col in LIST_LIKE_COLS:
        if col in extra_all.columns:
            extra_all[col] = extra_all[col].apply(
                lambda v: "\n".join(v) if isinstance(v, list) else v
            )

    # 2.3. обеспечиваем полный набор колонок и порядок, как в df_cv
    for col in df_cv.columns:
        if col not in extra_all.columns:
            extra_all[col] = pd.NA
    extra_all = extra_all[df_cv.columns]

    # --- 3. финальная склейка ----------------------------------------------------
    df_cv = pd.concat([df_cv, extra_all], ignore_index=True)
    st.success(f"Добавлено дополнительных резюме: {len(extra_all)}")

    return df_cv

def process_candidate_for_huntflow(candidate_key, candidate_ml_data):
    """
    Обрабатывает одного кандидата для отправки в Huntflow.
    Возвращает (True, "Сообщение об успехе") или (False, "Сообщение об ошибке").
    """
    session_key_pii = f"pii_data_{candidate_key}"
    session_key_hf_id = f"hf_app_id_{candidate_key}"
    session_key_hf_processed = f"hf_processed_{candidate_key}"
    
    cand_display_name = candidate_ml_data.get("Должность", f"Кандидат ID: {candidate_key}")
    
    try:
        pii_data_raw = None
        access_token_ext = None
        
        candidate_link = candidate_ml_data.get("link")
        source_type, resume_id_from_link = parse_resume_link(candidate_link)

        if not source_type or not resume_id_from_link:
            msg = f"Не удалось извлечь источник/ID из ссылки: {candidate_link}"
            st.warning(msg); logger.error(f"Парсинг ссылки для {cand_display_name}: {candidate_link}")
            return False, msg

        logger.info(f"Для {cand_display_name}: источник='{source_type}', ID='{resume_id_from_link}'")

        if source_type == "hh":
            access_token_ext = get_hh_oauth_token()
            # access_token_ext = st.secrets.get("HH_API_TOKEN")
            if access_token_ext: pii_data_raw = get_hh_contacts_api(resume_id_from_link, access_token_ext)
        elif source_type == "avito":
            access_token_ext = get_avito_oauth_token()
            if access_token_ext: pii_data_raw = get_avito_contacts_api(resume_id_from_link, access_token_ext)
        else:
            msg = f"Источник '{source_type}' не поддерживается."
            st.warning(msg); logger.warning(f"Неподдерживаемый источник '{source_type}' для {cand_display_name}")
            return False, msg
        
        if not access_token_ext: 
            msg = f"Нет Access Token для {source_type.upper()}."
            st.warning(msg); logger.error(f"Нет Access Token {source_type.upper()} для {cand_display_name}")
            return False, msg
        if not pii_data_raw: 
            msg = f"Не удалось получить ПД для кандидата из {source_type.upper()}."
            st.warning(msg); logger.error(f"Нет ПД {source_type.upper()} для {cand_display_name} (ID: {resume_id_from_link})")
            return False, msg
        
        pii_standardized = _extract_pii_details(pii_data_raw, source_type.upper())
        st.session_state[session_key_pii] = pii_standardized
        
        hf_app_id, err_create = create_huntflow_applicant_api(pii_standardized, candidate_ml_data, resume_id_from_link, source_type.upper())
        if err_create:
            msg = f"Ошибка создания кандидата в HF: {err_create}"
            st.warning(msg); logger.error(f"Ошибка создания {cand_display_name} в HF: {err_create}")
            return False, msg

        st.session_state[session_key_hf_id] = hf_app_id
        
        current_hf_vacancy_id = st.session_state.get("selected_huntflow_vacancy_id")
        if current_hf_vacancy_id:
            cand_score = candidate_ml_data.get("sim_score_second", candidate_ml_data.get("Итоговый балл",0))
            cand_score_perc = 0
            if isinstance(cand_score, (float, int)) and not isinstance(cand_score, bool):
                cand_score_perc = round(cand_score * 100 if 0 < cand_score <= 1.0 else cand_score)
            
            ok_link, err_link = link_applicant_to_vacancy_api(hf_app_id, current_hf_vacancy_id, cand_score_perc)
            if err_link:
                logger.warning(f"Ошибка привязки {cand_display_name} к вакансии HF {current_hf_vacancy_id}: {err_link}")
            else:
                logger.info(f"{cand_display_name} привязан к вакансии HF {current_hf_vacancy_id}.")
        
        st.session_state[session_key_hf_processed] = True
        success_msg = f"Кандидат успешно обработан для Huntflow (ID: {hf_app_id})."
        logger.info(f"{cand_display_name}: {success_msg}")
        return True, success_msg

    except Exception as e:
        error_msg = f"Непредвиденная ошибка при обработке для Huntflow: {e}"
        logger.error(f"{cand_display_name}: {error_msg}")
        return False, error_msg


def process_candidate_for_whatsapp(candidate_key, candidate_ml_data):
    """
    Отправляет сообщение кандидату через WhatsApp.
    Возвращает (True, "Сообщение об успехе") или (False, "Сообщение об ошибке").
    """
    session_key_pii = f"pii_data_{candidate_key}"
    session_key_hf_id = f"hf_app_id_{candidate_key}"
    session_key_wa_sent = f"wa_sent_{candidate_key}"
    
    # Проверяем, есть ли все необходимые данные
    if not st.session_state.get(session_key_hf_id) or not st.session_state.get(session_key_pii):
        msg = "Недостаточно данных для отправки в WhatsApp (требуется обработка в Huntflow)."
        st.warning(msg)
        return False, msg

    pii_for_whatsapp = st.session_state.get(session_key_pii)
    # Если это строка, пробуем распарсить как JSON
    if isinstance(pii_for_whatsapp, str):
        try:
            parsed = json.loads(pii_for_whatsapp)
            if isinstance(parsed, dict):
                pii_for_whatsapp = parsed
                st.session_state[session_key_pii] = parsed  # Сохраняем обратно, чтобы не парсить снова
            else:
                msg = f"Ошибка: данные кандидата для WhatsApp — строка, но не JSON-словарь: {pii_for_whatsapp}"
                st.warning(msg)
                logger.error(f"Chat2Desk: неверный формат pii_for_whatsapp для кандидата {candidate_key} - {pii_for_whatsapp}")
                return False, msg
        except Exception as e:
            msg = f"Ошибка: данные кандидата для WhatsApp — строка, не удалось распарсить как JSON: {pii_for_whatsapp}"
            st.warning(msg)
            logger.error(f"Chat2Desk: не удалось распарсить pii_for_whatsapp для кандидата {candidate_key}: {e} - {pii_for_whatsapp}")
            return False, msg
    if not isinstance(pii_for_whatsapp, dict):
        msg = f"Ошибка: данные кандидата для WhatsApp имеют неверный формат. {pii_for_whatsapp}"
        st.warning(msg)
        logger.error(f"Chat2Desk: неверный формат pii_for_whatsapp для кандидата {candidate_key} - {pii_for_whatsapp}")
        return False, msg
    phone = pii_for_whatsapp.get("phone")
    if not phone:
        msg = "Не найден номер телефона кандидата."
        st.warning(msg)
        logger.error(f"Chat2Desk: нет номера телефона для кандидата {candidate_key}")
        return False, msg

    try:
        applicant_id_for_whatsapp = st.session_state.get(session_key_hf_id)
        name_client = f"{pii_for_whatsapp.get('first_name', '')} {pii_for_whatsapp.get('last_name', '')}".strip()
        hf_account_id_for_c2d = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        current_hf_vacancy_id_for_c2d = st.session_state.get("selected_huntflow_vacancy_id")
        logger.success("Параметры извлечены из стейта!")

        vacancy_name_for_c2d = "Не указано"
        if current_hf_vacancy_id_for_c2d and st.session_state.get('huntflow_vacancies_details'):
            vac_json_path = st.session_state['huntflow_vacancies_details'].get(current_hf_vacancy_id_for_c2d)
            if vac_json_path and os.path.exists(vac_json_path):
                try:
                    with open(vac_json_path, "r", encoding="utf-8") as f:
                        vac_json = json.load(f)
                    vacancy_name_for_c2d = vac_json.get("position", "Не указано")
                except Exception as e:
                    logger.error(f"Ошибка чтения файла вакансии {vac_json_path}: {e}")
        
        success_c2d, msg_c2d = send_to_chat2desk_api(
            phone_number=phone,
            client_name=name_client,
            hf_account_id=hf_account_id_for_c2d,
            hf_vacancy_id=current_hf_vacancy_id_for_c2d,
            hf_applicant_id=applicant_id_for_whatsapp,
            vacancy_name_original=vacancy_name_for_c2d
        )
        
        if success_c2d:
            st.session_state[session_key_wa_sent] = True
            logger.info(f"Chat2Desk: Сообщение для {name_client} отправлено. {msg_c2d}")
            return True, msg_c2d
        else:
            st.warning(f"Chat2Desk: {msg_c2d}")
            logger.error(f"Chat2Desk: Ошибка отправки для {name_client}. {msg_c2d}")
            return False, msg_c2d

    except Exception as e:
        error_msg = f"Непредвиденная ошибка при отправке в WhatsApp: {e}"
        logger.error(f"Кандидат {candidate_key}: {error_msg}")
        return False, error_msg

# Загрузка вакансий Huntflow один раз при старте или если список пуст
if not st.session_state.huntflow_vacancies_list:
    with st.spinner("Загрузка активных вакансий из Huntflow..."):
        st.session_state.huntflow_vacancies_list, st.session_state.huntflow_vacancies_details = fetch_huntflow_vacancies()

# --- Отображение UI ---
common_ui.display_sidebar(config, update_address_weight_callback)

# --- Отображение формы ввода (АСПП или Проф) ---
vacancy_input_data = None
if current_mode == Mode.MASS:
    st.subheader("Параметры для подбора (АСПП)")
    vacancy_input_data = mass_ui.display_mass_input_form() 
elif current_mode == Mode.PROF:
    st.header("Профессиональный подбор кандидатов")
    vacancy_input_data = prof_ui.display_prof_input_form()

scrape_extra_resumes = st.checkbox(
    "произвести поиск доп. резюме в интернете (около 10 минут)",
    key="scrape_extra_resumes_flag"
)

if st.button("Подобрать", type="primary"):
    if not vacancy_input_data:
        st.warning("Данные вакансии не получены от формы ввода.")
    
    if not vacancy_input_data:
        st.warning("Режим подбора не вернул данные для вакансии. Возможно, он еще не реализован.")
    elif (
        vacancy_input_data.get("Должность")
        and vacancy_input_data.get("required")
        and vacancy_input_data.get("location")
    ):
        st.session_state["computed"] = False
        
        selected_date_threshold = st.session_state.selected_date_threshold
        threshold_from_slider = st.session_state.current_threshold 
        distance_option_val = st.session_state.distance_option
        is_vahta_val = st.session_state.vahta_mode
        
        if current_mode == Mode.PROF:
            st.info("Логика проф. подбора еще не полностью реализована.")
            st.stop() 
        else: # Mode.MASS
            df_cv_path = Path(config.get("data", {}).get("cv_path_mass", "data/processed/mass/candidates_hh.csv"))
            df_cv = load_data(str(df_cv_path))
            if df_cv.empty:
                st.error(f"Не удалось загрузить данные кандидатов из {df_cv_path}.")
                st.stop()

            with st.status("Подготовка вакансии..."):
                vacancy_processed = selector.preprocess_vacancy(deepcopy(vacancy_input_data)) 

            logger.info(f"Параметры вакансии:\n{vacancy_processed}")

            if scrape_extra_resumes:
                with st.spinner("Ищем дополнительные резюме в интернете (до 10 минут)…"):
                    # vacancy_input_data уже содержит актуальные поля формы;
                    # при желании можно заменить на st.session_state.selected_huntflow_vacancy_details
                    df_cv = append_extra_resumes(df_cv, vacancy_processed)
                
            if "address" in df_cv.columns: # Переименование, если есть
                 df_cv = df_cv.rename(columns={"address": "Адрес"})
        
        with st.status("Подбор кандидатов..."):
            use_cache = not (not Path("./tmp_cvs.csv").exists() or config["general"]["mode"] == "prod")
            
            if not use_cache:
                st.write(f"Первая фаза: анализ {df_cv.shape[0]} кандидатов...")
                max_distance_filter = None
                if distance_option_val != "Нет ограничений":
                    max_distance_filter = float(distance_option_val)

                current_first_stage_weights = deepcopy(config["model"]["stage_1"]["weights"]) 
                if is_vahta_val:
                    if len(current_first_stage_weights) > 1: 
                        current_first_stage_weights[1] = 0.0
                    else:
                        logger.warning("Не удалось обнулить вес 'Адрес' для 1-го этапа.")

                df_ranked_1st = selector.rank_first_stage(
                    vacancy=vacancy_processed, 
                    df_relevant=df_cv.copy(),
                    date_threshold=selected_date_threshold, 
                    is_vahta=is_vahta_val, 
                    max_distance_filter=max_distance_filter,
                    first_stage_weights=np.array(current_first_stage_weights),
                )
                st.write(f"Вторая фаза: анализ {df_ranked_1st.shape[0]} кандидатов..")

                df_ranked_2nd, vacancy_prep_for_display, nan_mask_for_display = selector.rank_second_stage(
                    vacancy=vacancy_processed,
                    df_relevant=df_ranked_1st.copy(),
                    df_weights=st.session_state["df_weights"], 
                    score_threshold_stage_2=threshold_from_slider,
                    top_n_second_stage = st.session_state.kols_candidates
                )
                if config["general"]["mode"] != "prod":
                    df_ranked_2nd.to_csv("./tmp_cvs.csv", index=False)
                    with open("./tmp_vac.json", "w", encoding='utf-8') as f:
                        json.dump(vacancy_prep_for_display, f, ensure_ascii=False)
                    np.save("./tmp_mask.npy", nan_mask_for_display)
            else:
                st.write("Загрузка из кэша...")
                df_ranked_2nd = pd.read_csv("./tmp_cvs.csv")
                with open("./tmp_vac.json", "r", encoding='utf-8') as f:
                    vacancy_prep_for_display = json.load(f)
                nan_mask_for_display = np.load("./tmp_mask.npy")

            st.session_state.data_cv_dict = df2dict(df_ranked_2nd) 
            st.session_state.vacancy_prep_for_display = vacancy_prep_for_display
            st.session_state.nan_mask_for_display = nan_mask_for_display
            st.session_state["computed"] = True
            st.write(f"Выбрано {df_ranked_2nd.shape[0]} лучших кандидатов.")
            st.rerun() # Перезапускаем, чтобы отобразить результаты ниже кнопки
            
    else:
        st.error(
            "Обязательные поля для заполнения: 'Должность', 'Обязательные требования', 'Локация и формат работы'",
            icon="🚨",
        )
if st.session_state.get("computed", False):
    data_cv_to_display = st.session_state.get("data_cv_dict", {})
    vacancy_prep_to_display = st.session_state.get("vacancy_prep_for_display", {})
    nan_mask_to_display = st.session_state.get("nan_mask_for_display", np.array([]))

    if not data_cv_to_display:
        st.info("Нет кандидатов для отображения.")
    else:
        # Отображение результатов в зависимости от режима (как и было)
        if current_mode == Mode.MASS:
            if "model" in config and "stage_2" in config["model"] and "ranking_features" in config["model"]["stage_2"]:
                 mass_ui.display_mass_results(data_cv_to_display, vacancy_prep_to_display, config, nan_mask_to_display)
            else:
                st.error("Ошибка в структуре конфигурационного файла для отображения результатов.")
        elif current_mode == Mode.PROF:
            prof_ui.display_prof_results(data_cv_to_display, vacancy_prep_to_display, config, nan_mask_to_display)
        
        st.markdown("---") 
        st.subheader("Групповые действия с кандидатами")
        
        # --- НОВЫЕ ГРУППОВЫЕ КНОПКИ ---
        cols_actions = st.columns(2)
        with cols_actions[0]:
            hf_mass_button = st.button("✅ Отправить выбранных в Huntflow", use_container_width=True)
        with cols_actions[1]:
            wa_mass_button = st.button("🔵 Отправить выбранным в Whatsapp", use_container_width=True)
        
        st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'/>", unsafe_allow_html=True)
        
        # Список для хранения ключей выбранных кандидатов
        selected_candidates_keys = []

        # --- СПИСОК КАНДИДАТОВ С ЧЕКБОКСАМИ ---
        for candidate_key, candidate_ml_data in data_cv_to_display.items():
            if not isinstance(candidate_ml_data, dict):
                logger.warning(f"Данные для кандидата {candidate_key} не являются словарем, пропуск.")
                continue

            # Ключи для session_state (статусы обработки)
            session_key_hf_processed = f"hf_processed_{candidate_key}"
            session_key_wa_sent = f"wa_sent_{candidate_key}"
            
            cols_info = st.columns([1, 10]) # Колонка для чекбокса и для информации
            
            with cols_info[0]:
                # Чекбокс для выбора кандидата. Отключаем, если уже обработан в HF.
                is_processed_in_hf = st.session_state.get(session_key_hf_processed, False)
                if st.checkbox("", key=f"select_{candidate_key}", value=False, help="Выбрать кандидата для группового действия"):
                    selected_candidates_keys.append(candidate_key)

            with cols_info[1]:
                # Формирование имени для отображения
                cand_display_name = candidate_ml_data.get("Должность", f"Кандидат ID: {candidate_key}")
                
                cand_score = candidate_ml_data.get("sim_score_second", candidate_ml_data.get("Итоговый балл",0))
                cand_score_perc = "N/A"
                if isinstance(cand_score, (float, int)) and not isinstance(cand_score, bool):
                    cand_score_perc = round(cand_score * 100 if 0 < cand_score <= 1.0 else cand_score)
                
                # Отображение статусов
                status_icons = []
                if st.session_state.get(session_key_hf_processed, False):
                    status_icons.append("✅")
                if st.session_state.get(session_key_wa_sent, False):
                    status_icons.append("🔵")
                
                st.markdown(f"{' '.join(status_icons)} {cand_display_name} (Скоринг: {cand_score_perc}%)")
            
            st.markdown("<hr style='margin-top:0.1rem; margin-bottom:0.1rem; border-top: 1px dashed #222;'/>", unsafe_allow_html=True)

        # --- ЛОГИКА ОБРАБОТКИ ГРУППОВЫХ ДЕЙСТВИЙ ---

        # Если нажата кнопка "В Huntflow"
        if hf_mass_button:
            if not selected_candidates_keys:
                st.warning("Пожалуйста, выберите хотя бы одного кандидата с помощью галочки.")
            else:
                processed_count = 0
                with st.status(f"Обработка {len(selected_candidates_keys)} кандидатов для Huntflow...", expanded=True) as status:
                    for key in selected_candidates_keys:
                        # Пропускаем уже обработанных
                        if st.session_state.get(f"hf_processed_{key}", False):
                            st.write(f"ℹ️ Кандидат {key} уже был обработан ранее, пропуск.")
                            continue

                        st.write(f"▶️ Обработка кандидата {key}...")
                        success, message = process_candidate_for_huntflow(key, data_cv_to_display[key])
                        if success:
                            st.write(f"✔️ {message}")
                            processed_count += 1
                        else:
                            st.write(f"❌ {message}")
                    
                    status.update(label=f"Обработка для Huntflow завершена! Успешно: {processed_count} из {len(selected_candidates_keys)}.", state="complete")
                st.rerun() # Перезапускаем скрипт, чтобы обновить UI (иконки статусов)

        # Если нажата кнопка "В Whatsapp"
        if wa_mass_button:
            if not selected_candidates_keys:
                st.warning("Пожалуйста, выберите хотя бы одного кандидата с помощью галочки.")
            else:
                sent_count = 0
                with st.status(f"Отправка сообщений {len(selected_candidates_keys)} кандидатам...", expanded=True) as status:
                    for key in selected_candidates_keys:
                        # Пропускаем тех, кому уже отправляли
                        if st.session_state.get(f"wa_sent_{key}", False):
                            st.write(f"ℹ️ Кандидату {key} сообщение уже отправлялось, пропуск.")
                            continue

                        st.write(f"▶️ Отправка сообщения кандидату {key}...")
                        success, message = process_candidate_for_whatsapp(key, data_cv_to_display[key])
                        if success:
                            st.write(f"✔️ Сообщение отправлено. Ответ API: {message}")
                            sent_count += 1
                        else:
                            st.write(f"❌ Ошибка отправки: {message}")

                    status.update(label=f"Отправка в Whatsapp завершена! Успешно: {sent_count} из {len(selected_candidates_keys)}.", state="complete")
                st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'/>", unsafe_allow_html=True) # Горизонтальная линия между кандидатами