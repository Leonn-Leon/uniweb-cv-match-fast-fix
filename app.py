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

HUNTFLOW_BASE_URL = "https://rusal-api-mirror.huntflow.ru/v2" 

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

def fetch_huntflow_vacancies_api():
    """Загружает активные вакансии из Huntflow."""
    proxies = get_huntflow_proxies()
    try:
        hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
        hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        if not hf_token:
            st.error("Токен API Huntflow не найден в секретах. Проверьте .streamlit/secrets.toml")
            return [], {}
        headers = {"Authorization": f"Bearer {hf_token}"}
        url = f"{HUNTFLOW_BASE_URL}/accounts/{hf_account_id}/vacancies"
        params = {"status": "OPEN", "count": 100}
        response = requests.get(url, headers=headers, params=params, proxies=proxies)
        response.raise_for_status()
        vacancies_data = response.json().get("items", [])
        vac_list_selectbox = [(vac.get("position", f"Вакансия ID {vac.get('id')}"), vac.get("id")) for vac in vacancies_data if vac.get("id")]
        vac_details_map = {vac.get("id"): vac for vac in vacancies_data if vac.get("id")}
        return vac_list_selectbox, vac_details_map
    except requests.exceptions.ProxyError as e:
        st.error(f"Ошибка прокси при подключении к Huntflow: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке вакансий из Huntflow: {e}")
    except Exception as e:
        st.error(f"Неожиданная ошибка (вакансии Huntflow): {e}")
    return [], {}

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
        if not r_contacts:
            contacts_href = r_resume.json().get("actions").get('get_with_contact').get("url")
            if not contacts_href:
                st.warning(f"Нет ссылки на контакты HH {resume_id}."); return None
            r_contacts = requests.get(contacts_href, headers=headers)
            r_contacts.raise_for_status()
            r_contacts = r_contacts.json().get("contact")
        details = {"first_name": r_resume.json().get("first_name", ""), "last_name": r_resume.json().get("last_name", ""),
                   "middle_name": r_resume.json().get("middle_name", ""), "phone": "", "email": ""}
        if r_contacts[0].get("type").get("id") == "email":
            details["email"] = r_contacts[0].get("value", "")
            details["phone"] = r_contacts[1].get("value", "").get("formatted")
        elif r_contacts[0].get("type").get("id") == "cell":
            details["phone"] = r_contacts[0].get("value").get("formatted")
            details["email"] = r_contacts[1].get("value")
        return details
    except requests.exceptions.RequestException as e: st.error(f"Ошибка API HH.ru ({resume_id}): {e}"); return None
    except Exception as e: st.error(f"Ошибка (HH contacts {resume_id}): {e}"); return None

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

HH_ACCESS_TOKEN_KEY = "hh_access_token_val" # Изменил имя ключа, чтобы не пересекаться с предыдущими
HH_TOKEN_EXPIRES_AT_KEY = "hh_token_expires_at_val"

def get_hh_oauth_token():
    access_token = st.session_state.get(HH_ACCESS_TOKEN_KEY)
    expires_at = st.session_state.get(HH_TOKEN_EXPIRES_AT_KEY)

    if access_token and expires_at and time.time() < expires_at:
        logger.debug("HH: Используется существующий валидный токен.")
        return access_token

    logger.info("HH: Попытка получения нового access token...")
    token_url = st.secrets.get("HH_TOKEN_URL")
    client_id = st.secrets.get("HH_CLIENT_ID")
    client_secret = st.secrets.get("HH_CLIENT_SECRET")

    if not all([token_url, client_id, client_secret]):
        st.error("HH.ru: CLIENT_ID, CLIENT_SECRET или TOKEN_URL не настроены в секретах.")
        logger.error("HH.ru: CLIENT_ID, CLIENT_SECRET или TOKEN_URL не настроены в секретах.")
        return None

    payload = {
        'grant_type': 'client_credentials', # Это стандарт для client_credentials
        'client_id': client_id,       # HH может требовать client_id/secret в теле
        'client_secret': client_secret  # или через Basic Auth. Нужно проверить документацию!
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
        response = requests.post(token_url, data=payload, headers=headers)
        response.raise_for_status() # Проверка на HTTP ошибки
        token_data = response.json()
        
        new_access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 86400) # Время жизни в секундах, по дефолту сутки (86400 сек)

        if new_access_token and isinstance(expires_in, int):
            st.session_state[HH_ACCESS_TOKEN_KEY] = new_access_token
            st.session_state[HH_TOKEN_EXPIRES_AT_KEY] = time.time() + expires_in - 60 # Буфер 60 сек
            st.success("HH.ru: Access token успешно получен/обновлен.")
            logger.info("HH.ru: Access token успешно получен/обновлен.")
            return new_access_token
        else:
            st.error(f"HH.ru: Не удалось получить 'access_token' или 'expires_in' из ответа. Ответ: {token_data}")
            logger.error(f"HH.ru: Не удалось получить 'access_token' или 'expires_in' из ответа. Ответ: {token_data}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"HH.ru: Ошибка при получении токена: {e}")
        logger.error(f"HH.ru: Ошибка при получении токена: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                st.json(e.response.json())
                logger.error(f"HH.ru: Тело ошибки: {e.response.json()}")
            except ValueError:
                st.text(e.response.text)
                logger.error(f"HH.ru: Тело ошибки (не JSON): {e.response.text}")
        return None

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

    logger.info(f"Chat2Desk: Запрос на URL 1: {url_with_params}")

    params_for_url = {
        "code": chat2desk_code,
        "params[phonenumber]": 77770444258,
        "params[nameclient]": "Вячеслав",
        "params[account_id]": str(2),
        "params[vacancy_id]": str(24295) if hf_vacancy_id else "",
        "params[application_id]": str(469314),
        "params[vacancyname]": "Эксперт"
    }
    
    param_pairs = [f"{quote(key)}={quote(str(value))}" for key, value in params_for_url.items()]
    url_with_params = f"{CHAT2DESK_BASE_URL}?{'&'.join(param_pairs)}"

    logger.info(f"Chat2Desk: Запрос на URL: {url_with_params}")

    try:
        response = requests.post(url_with_params, headers={'Content-Type': 'application/json'}) 
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
# Загрузка вакансий Huntflow один раз при старте или если список пуст
if not st.session_state.huntflow_vacancies_list:
    with st.spinner("Загрузка активных вакансий из Huntflow..."):
        st.session_state.huntflow_vacancies_list, st.session_state.huntflow_vacancies_details = fetch_huntflow_vacancies_api()

# --- Отображение UI ---
common_ui.display_sidebar(config, update_address_weight_callback)

# --- БЛОК: Выбор вакансии из Huntflow и форма ввода ---
st.header("Выбор вакансии и ввод данных для подбора")
if st.session_state.huntflow_vacancies_list:
    vacancy_options = ["-- Ввести вручную --"] + [name for name, _ in st.session_state.huntflow_vacancies_list]
    selected_vacancy_display_name = st.selectbox(
        "Выберите вакансию из Huntflow (для предзаполнения полей ниже):",
        options=vacancy_options,
        index=0, # По умолчанию "-- Ввести вручную --"
        key="hf_vacancy_selectbox_selector" 
    )

    if selected_vacancy_display_name != "-- Ввести вручную --":
        selected_idx_in_hf_list = next((i for i, (name, _id) in enumerate(st.session_state.huntflow_vacancies_list) if name == selected_vacancy_display_name), -1)

        if selected_idx_in_hf_list != -1:
            newly_selected_hf_id = st.session_state.huntflow_vacancies_list[selected_idx_in_hf_list][1]
            if st.session_state.selected_huntflow_vacancy_id != newly_selected_hf_id:
                st.session_state.selected_huntflow_vacancy_id = newly_selected_hf_id
                details = st.session_state.huntflow_vacancies_details.get(newly_selected_hf_id)
                logger.debug(f"Выбрана вакансия c ID: {newly_selected_hf_id}, её значение: {details}")

                if details:
                    # 1. Должность
                    st.session_state.vacancy_form_title = details.get("position", "")
                    
                    required_texts = []
                    education_req = details.get("education")
                    if education_req and isinstance(education_req, str):
                        required_texts.append(f"Образование: {education_req.strip()}")
                    
                    experience_req = details.get("experience_position")
                    if experience_req and isinstance(experience_req, str):
                        required_texts.append(f"Требования к опыту: {experience_req.strip()}")

                    if required_texts:
                        st.session_state.vacancy_form_required = "\n".join(required_texts)
                    else:
                        st.session_state.vacancy_form_required = "" 
                        logger.warning(f"Не удалось извлечь структурированные обязательные требования для вакансии ID {newly_selected_hf_id}. Поля 'education' и 'experience_position' пусты или не строки.")

                    location_format_texts = []
                    region_rusal_id = details.get("region_rusal")
                    if region_rusal_id and isinstance(region_rusal_id, int):
                        location_format_texts.append("Локация: " + get_region_name_by_id(region_rusal_id))

                    money_info = details.get("money")
                    if money_info and isinstance(money_info, str):
                        location_format_texts.append("Зарплата: " + money_info.strip())
                    
                    # Формат работы можно попробовать извлечь из "contract"
                    contract_type = details.get("contract")
                    if contract_type and isinstance(contract_type, str):
                        location_format_texts.append(f"Тип договора: {contract_type.strip()}")

                    # Готовность к командировкам
                    business_trip_info = details.get("business_trip")
                    if business_trip_info and isinstance(business_trip_info, str):
                        location_format_texts.append(f"Командировки: {business_trip_info}")
                    elif isinstance(business_trip_info, bool):
                        location_format_texts.append(f"Командировки: {'Да' if business_trip_info else 'Нет'}")

                    if location_format_texts:
                        st.session_state.vacancy_form_location = "\n".join(location_format_texts)
                    else:
                        st.session_state.vacancy_form_location = ""
                        logger.warning(f"Не удалось извлечь информацию о локации/формате работы для вакансии ID {newly_selected_hf_id}.")

                    
                    optional_texts = []

                    benefits_info = details.get("benefits") # В примере None
                    if benefits_info and isinstance(benefits_info, str):
                        optional_texts.append(f"Условия: {benefits_info.strip()}")
                    
                    notes_info = details.get("notes") # В примере None
                    if notes_info and isinstance(notes_info, str):
                        optional_texts.append(f"Заметки: {notes_info.strip()}")
                    
                    if optional_texts:
                        st.session_state.vacancy_form_optional = "\n".join(optional_texts)
                    else:
                        st.session_state.vacancy_form_optional = ""

                    st.toast(f"Поля предзаполнены вакансией: {details.get('position')}", icon="ℹ️")
                    st.rerun() # Перезапускаем, чтобы виджеты обновились
                else: # Если details не найдены (хотя не должно быть, если ID есть в списке)
                    logger.error(f"Детали для вакансии ID {newly_selected_hf_id} не найдены в st.session_state.huntflow_vacancies_details")
                    # Можно сбросить поля или ничего не делать
                    st.session_state.vacancy_form_title = "" 
                    st.session_state.vacancy_form_required = ""
                    st.session_state.vacancy_form_location = ""
                    st.session_state.vacancy_form_optional = ""
                    # st.rerun() # Если нужно, чтобы сброс отразился

        else: 
            if st.session_state.selected_huntflow_vacancy_id is not None: # Сбрасываем, если была выбрана, а теперь не найдена
                st.session_state.selected_huntflow_vacancy_id = None
                # Очистка полей
                st.session_state.vacancy_form_title = "" 
                st.session_state.vacancy_form_required = ""
                st.session_state.vacancy_form_location = ""
                st.session_state.vacancy_form_optional = ""
                st.warning(f"Не удалось найти детали для выбранной вакансии '{selected_vacancy_display_name}'. Поля сброшены.")
                st.rerun() # Перезапускаем, чтобы сброс отразился

    elif selected_vacancy_display_name == "-- Ввести вручную --" and st.session_state.selected_huntflow_vacancy_id is not None:
        st.session_state.selected_huntflow_vacancy_id = None 
        st.session_state.vacancy_form_title = "" 
        st.session_state.vacancy_form_required = ""
        st.session_state.vacancy_form_location = ""
        st.session_state.vacancy_form_optional = ""
        st.toast("Поля сброшены для ввода вручную.", icon="✍️")
        st.rerun()

# --- Отображение формы ввода (АСПП или Проф) ---
vacancy_input_data = None
if current_mode == Mode.MASS:
    st.subheader("Параметры для подбора (АСПП)")
    vacancy_input_data = mass_ui.display_mass_input_form() 
elif current_mode == Mode.PROF:
    st.header("Профессиональный подбор кандидатов")
    vacancy_input_data = prof_ui.display_prof_input_form()

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
            if "address" in df_cv.columns: # Переименование, если есть
                 df_cv = df_cv.rename(columns={"address": "Адрес"})

        with st.status("Подготовка вакансии..."):
            vacancy_processed = selector.preprocess_vacancy(deepcopy(vacancy_input_data)) 
        
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
        if current_mode == Mode.MASS:
            if "model" in config and "stage_2" in config["model"] and "ranking_features" in config["model"]["stage_2"]:
                 mass_ui.display_mass_results(
                    data_cv_to_display, 
                    vacancy_prep_to_display, 
                    config, 
                    nan_mask_to_display
                )
            else:
                st.error("Ошибка в структуре конфигурационного файла для отображения результатов.")
        elif current_mode == Mode.PROF:
            prof_ui.display_prof_results(
                data_cv_to_display, 
                vacancy_prep_to_display, 
                config, 
                nan_mask_to_display
            )
        
        
        st.markdown("---") 
        st.subheader("Действия с отобранными кандидатами")

        if not data_cv_to_display:
            st.info("Нет кандидатов для отображения действий.")
        else:
            for candidate_key, candidate_ml_data in data_cv_to_display.items():
                if not isinstance(candidate_ml_data, dict):
                    logger.warning(f"Данные для кандидата {candidate_key} не являются словарем, пропуск.")
                    continue

                # Ключи для session_state для этого кандидата
                session_key_hf_id = f"hf_app_id_{candidate_key}"
                session_key_pii = f"pii_data_{candidate_key}"
                session_key_hf_processed = f"hf_processed_{candidate_key}" # Флаг, что кандидат в HF

                # Отображение информации о кандидате
                cols_info_buttons = st.columns([3, 1, 1]) # Одна колонка для инфо, две для кнопок
                with cols_info_buttons[0]:
                    cand_display_name = candidate_ml_data.get("Должность", f"Кандидат ID: {candidate_key}")
                    if st.session_state.get(session_key_pii, {}).get('first_name') != 'Неизв.': # Если имя уже раскрыто
                        cand_display_name = f"{st.session_state.get(session_key_pii, {}).get('first_name', '')} {st.session_state.get(session_key_pii, {}).get('last_name', '')} ({cand_display_name})"
                    elif candidate_ml_data.get("ФИО"): # Если есть в исходных данных
                         cand_display_name = f"{candidate_ml_data.get('ФИО')} ({cand_display_name})"
                    
                    cand_score = candidate_ml_data.get("sim_score_second", candidate_ml_data.get("Итоговый балл",0))
                    cand_score_perc = "N/A"
                    if isinstance(cand_score, (float, int)) and not isinstance(cand_score, bool):
                        cand_score_perc = round(cand_score * 100 if 0 < cand_score <= 1.0 else cand_score)
                    
                    st.markdown(f"{cand_display_name} (Скоринг: {cand_score_perc}%)")
                
                # Кнопка "В Huntflow"
                with cols_info_buttons[1]:
                    hf_button_key = f"hf_btn_{candidate_key}"
                    # Блокируем кнопку, если уже успешно обработан
                    disable_hf_button = st.session_state.get(session_key_hf_processed, False) 
                    
                    if st.button("В Huntflow", key=hf_button_key, help=f"Раскрыть ПД и создать {cand_display_name} в Huntflow", disabled=disable_hf_button):
                        st.markdown(f"--- \n_Обработка для Huntflow: {cand_display_name}..._")
                        with st.spinner(f"Работаем с {cand_display_name}..."):
                            pii_data_raw = None
                            access_token_ext = None
                            
                            candidate_link = candidate_ml_data.get("link")
                            source_type, resume_id_from_link = parse_resume_link(candidate_link)

                            if not source_type or not resume_id_from_link:
                                st.error(f"Не удалось извлечь источник/ID из ссылки: {candidate_link}"); logger.error(f"Парсинг ссылки для {cand_display_name}: {candidate_link}"); continue 
                            logger.info(f"Для {cand_display_name}: источник='{source_type}', ID='{resume_id_from_link}'")

                            if source_type == "hh":
                                # access_token_ext = get_hh_oauth_token()
                                access_token_ext = st.secrets.get("HH_API_TOKEN")
                                if access_token_ext: pii_data_raw = get_hh_contacts_api(resume_id_from_link, access_token_ext)
                            elif source_type == "avito":
                                access_token_ext = get_avito_oauth_token()
                                if access_token_ext: pii_data_raw = get_avito_contacts_api(resume_id_from_link, access_token_ext)
                            else:
                                st.warning(f"Источник '{source_type}' не поддерживается."); logger.warning(f"Неподдерживаемый источник '{source_type}' для {cand_display_name}"); continue
                            
                            if not access_token_ext: st.error(f"Нет Access Token для {source_type.upper()}."); logger.error(f"Нет Access Token {source_type.upper()} для {cand_display_name}"); continue
                            if not pii_data_raw: st.error(f"Нет ПД для {cand_display_name} из {source_type.upper()}."); logger.error(f"Нет ПД {source_type.upper()} для {cand_display_name} (ID: {resume_id_from_link})"); continue
                            
                            st.success(f"ПД для {cand_display_name} из {source_type.upper()} получены."); logger.info(f"ПД для {cand_display_name} ({source_type.upper()}) получены.")
                            pii_standardized = _extract_pii_details(pii_data_raw, source_type.upper())
                            st.session_state[session_key_pii] = pii_standardized # <-- Сохраняем PII
                            st.write("Извлеченные ПД:"); st.json(pii_standardized) # Для отладки

                            hf_app_id, err_create = create_huntflow_applicant_api(pii_standardized, candidate_ml_data, resume_id_from_link, source_type.upper())
                            if err_create: st.error(f"Ошибка создания {cand_display_name} в HF: {err_create}"); logger.error(f"Ошибка создания {cand_display_name} в HF: {err_create}"); continue
                            st.success(f"Кандидат {cand_display_name} создан в HF (ID: {hf_app_id})."); logger.info(f"Кандидат {cand_display_name} создан в HF (ID: {hf_app_id}).")
                            st.session_state[session_key_hf_id] = hf_app_id # <-- Сохраняем ID аппликанта HF
                            
                            # ok_q, err_q = fill_huntflow_questionary_api(hf_app_id, candidate_ml_data) # Анкета
                            # if err_q: st.warning(f"Ошибка анкеты {cand_display_name} в HF: {err_q}"); logger.warning(f"Ошибка анкеты {cand_display_name} в HF: {err_q}")
                            # else: st.info(f"Анкета {cand_display_name} в HF обработана."); logger.info(f"Анкета {cand_display_name} в HF обработана.")

                            current_hf_vacancy_id = st.session_state.get("selected_huntflow_vacancy_id") # Привязка к вакансии
                            if current_hf_vacancy_id:
                                ok_link, err_link = link_applicant_to_vacancy_api(hf_app_id, current_hf_vacancy_id, cand_score_perc if cand_score_perc != "N/A" else 0)
                                if err_link: st.warning(f"Ошибка привязки {cand_display_name} к вакансии HF {current_hf_vacancy_id}: {err_link}"); logger.warning(f"Ошибка привязки {cand_display_name} к вакансии HF {current_hf_vacancy_id}: {err_link}")
                                else: st.info(f"{cand_display_name} привязан к вакансии HF {current_hf_vacancy_id}."); logger.info(f"{cand_display_name} привязан к вакансии HF {current_hf_vacancy_id}.")
                            else: st.info(f"{cand_display_name} не привязан (вакансия HF не выбрана)."); logger.info(f"{cand_display_name} не привязан к вакансии HF (не выбрана).")
                            
                            st.session_state[session_key_hf_processed] = True # Помечаем как полностью обработанного для HF
                            st.success(f"Кандидат {cand_display_name} полностью обработан для Huntflow!")
                            logger.info(f"Кандидат {cand_display_name} полностью обработан для Huntflow!")
                            st.markdown("---")
                            # st.rerun() # Перерисовать UI
                
                # Кнопка "Связаться по Whatsapp"
                with cols_info_buttons[2]:
                    whatsapp_button_key = f"whatsapp_btn_{candidate_key}"
                    # Кнопка активна, если есть hf_app_id (т.е. кандидат создан в Huntflow) и есть ПД
                    can_send_whatsapp = st.session_state.get(session_key_hf_id) and st.session_state.get(session_key_pii)
                    
                    if st.button("Whatsapp", key=whatsapp_button_key, help=f"Связаться с {cand_display_name} по Whatsapp", disabled=not can_send_whatsapp):
                        pii_for_whatsapp = st.session_state.get(session_key_pii)
                        applicant_id_for_whatsapp = st.session_state.get(session_key_hf_id)
                        
                        phone = pii_for_whatsapp.get("phone")
                        name_client = f"{pii_for_whatsapp.get('first_name', '')} {pii_for_whatsapp.get('last_name', '')}".strip()
                        hf_account_id_for_c2d = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2") # Берем из секретов
                        current_hf_vacancy_id_for_c2d = st.session_state.get("selected_huntflow_vacancy_id")
                        
                        # Получаем название текущей выбранной вакансии из Huntflow для параметра vacancyname
                        vacancy_name_for_c2d = "Не указано"
                        if current_hf_vacancy_id_for_c2d and st.session_state.get('huntflow_vacancies_details'):
                            selected_vac_details = st.session_state['huntflow_vacancies_details'].get(current_hf_vacancy_id_for_c2d)
                            if selected_vac_details:
                                vacancy_name_for_c2d = selected_vac_details.get("position", "Не указано")
                        
                        if not phone:
                            st.error("Не найден номер телефона кандидата для отправки в Whatsapp.")
                            logger.error(f"Chat2Desk: нет номера телефона для {cand_display_name}")
                        elif not applicant_id_for_whatsapp: # Доп. проверка, хотя disabled должен был сработать
                            st.error("Не найден ID аппликанта в Huntflow.")
                            logger.error(f"Chat2Desk: нет ID аппликанта HF для {cand_display_name}")
                        else:
                            st.info(f"Попытка отправить сообщение {name_client} по номеру {phone} через Chat2Desk...")
                            with st.spinner(f"Отправка в Chat2Desk для {name_client}..."):
                                success_c2d, msg_c2d = send_to_chat2desk_api(
                                    phone_number=phone,
                                    client_name=name_client,
                                    hf_account_id=hf_account_id_for_c2d,
                                    hf_vacancy_id=current_hf_vacancy_id_for_c2d,
                                    hf_applicant_id=applicant_id_for_whatsapp,
                                    vacancy_name_original=vacancy_name_for_c2d
                                )
                                if success_c2d:
                                    st.success(f"Chat2Desk: {msg_c2d}")
                                else:
                                    st.error(f"Chat2Desk: {msg_c2d}")
                st.markdown("<hr style='margin-top:0.5rem; margin-bottom:0.5rem;'/>", unsafe_allow_html=True) # Горизонтальная линия между кандидатами