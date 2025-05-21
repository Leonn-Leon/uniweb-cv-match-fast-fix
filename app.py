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
        # response = requests.get(url, headers=headers, params=params)
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
        resume_url = f"https://api.hh.ru/resumes/{resume_id}"
        r_resume = requests.get(resume_url, headers=headers)
        # logger.debug(f"Получены данные:"+ str(r_resume.json()))
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
        details = {"first_name": FIO.get("first_name", ""), "last_name": FIO.get("last_name", ""),
                   "middle_name": FIO.get("patronymic", ""), "phone": "", "email": ""}
        r_contacts = response.json().get("contacts")
        if r_contacts[0].get("type") == "e-mail":
            details["email"] = r_contacts[0].get("value", "")
            details["phone"] = r_contacts[1].get("value", "")
        elif r_contacts[0].get("type") == "phone":
            details["phone"] = r_contacts[0].get("value")
            details["email"] = r_contacts[1].get("value")
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
            "position": candidate_ml_data.get("Должность", candidate_ml_data.get("Желаемая должность", "")),
            "money": str(candidate_ml_data.get("Зарплата от", "")),
            "externals": [{"auth_type": "NATIVE", "id": source_resume_id, 
                           "data": {"body": f"Кандидат из {source_type_for_hf}. Скоринг: {candidate_ml_data.get('Итоговый балл', 'N/A')}%"} }]
        }
        response = requests.post(url, headers=headers, json=body, proxies=proxies)
        response.raise_for_status()
        return response.json().get("id"), None
    except requests.exceptions.RequestException as e: return None, f"API HF (создание): {e.response.text if e.response else e}"
    except Exception as e: return None, f"Ошибка HF (создание): {e}"

def fill_huntflow_questionary_api(hf_applicant_id, candidate_ml_data):
    """Заполняет анкету кандидата в Huntflow (шаг 2, упрощенно)."""
    try:
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
        response = requests.post(url, headers=headers, json=body_filtered)
        response.raise_for_status()
        return True, None
    except requests.exceptions.RequestException as e: return False, f"API HF (анкета): {e.response.text if e.response else e}"
    except Exception as e: return False, f"Ошибка HF (анкета): {e}"

def link_applicant_to_vacancy_api(hf_applicant_id, hf_vacancy_id, score_percentage):
    """Привязывает кандидата к вакансии в Huntflow."""
    try:
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
        response = requests.post(url, headers=headers, json=body)
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
                logger.debug(f"Выбрана вакансия c ID: {newly_selected_hf_id}, её значение: {details}") # Оставил ваш логгер

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
                        st.session_state.vacancy_form_required = "\n\n".join(required_texts)
                    else:
                        st.session_state.vacancy_form_required = "" 
                        logger.warning(f"Не удалось извлечь структурированные обязательные требования для вакансии ID {newly_selected_hf_id}. Поля 'education' и 'experience_position' пусты или не строки.")

                    location_format_texts = []
                    region_rusal_id = details.get("region_rusal")
                    if region_rusal_id:
                        location_format_texts.append(f"Регион Русал (ID): {region_rusal_id}")
                    
                    # Формат работы можно попробовать извлечь из "contract"
                    contract_type = details.get("contract")
                    if contract_type and isinstance(contract_type, str):
                        location_format_texts.append(f"Тип договора: {contract_type.strip()}")

                    # Готовность к командировкам
                    business_trip_info = details.get("business_trip") # В примере None
                    if business_trip_info and isinstance(business_trip_info, str): # или если это boolean
                        location_format_texts.append(f"Командировки: {business_trip_info}")
                    elif isinstance(business_trip_info, bool):
                        location_format_texts.append(f"Командировки: {'Да' if business_trip_info else 'Нет'}")


                    if location_format_texts:
                        st.session_state.vacancy_form_location = "\n".join(location_format_texts)
                    else:
                        st.session_state.vacancy_form_location = ""
                        logger.warning(f"Не удалось извлечь информацию о локации/формате работы для вакансии ID {newly_selected_hf_id}.")

                    # 4. Дополнительные требования
                    #    Сюда можно поместить 
                    #    Поле "status" содержит много информации, включая HTML.
                    #    Поле "notes" (в примере None).
                    #    Поля "language_main", "proficiency_level_main".
                    #    Поле "benefits" (в примере None).
                    
                    optional_texts = []

                    language_main = details.get("language_main")
                    proficiency_level_main = details.get("proficiency_level_main")
                    if language_main and isinstance(language_main, str) and language_main.lower() != 'не требуется':
                        lang_text = f"Основной язык: {language_main.strip()}"
                        if proficiency_level_main and isinstance(proficiency_level_main, str):
                            lang_text += f" (Уровень: {proficiency_level_main.strip()})"
                        optional_texts.append(lang_text)

                    benefits_info = details.get("benefits") # В примере None
                    if benefits_info and isinstance(benefits_info, str):
                        optional_texts.append(f"Условия и бенефиты: {benefits_info.strip()}")
                    
                    notes_info = details.get("notes") # В примере None
                    if notes_info and isinstance(notes_info, str):
                        optional_texts.append(f"Заметки: {notes_info.strip()}")

                    # Поле "status" содержит HTML, его нужно очищать.
                    # status_html = details.get("status")
                    # if status_html:
                    #     from bs4 import BeautifulSoup # Потребует установки: pip install beautifulsoup4
                    #     soup = BeautifulSoup(status_html, "html.parser")
                    #     status_text = soup.get_text(separator="\n").strip()
                    #     if status_text: # Добавляем только если после очистки что-то осталось
                    #          optional_texts.append(f"Дополнительная информация (из status):\n{status_text}")
                    
                    if optional_texts:
                        st.session_state.vacancy_form_optional = "\n\n".join(optional_texts)
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
                    score_threshold_stage_2=threshold_from_slider 
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

        if not data_cv_to_display: # Добавил проверку, если data_cv_to_display пуст
            st.info("Нет кандидатов для отображения действий.")
        else:
            for candidate_key, candidate_ml_data in data_cv_to_display.items(): # Переименовал candidate_data в candidate_ml_data для ясности
                if not isinstance(candidate_ml_data, dict):
                    logger.warning(f"Данные для кандидата {candidate_key} не являются словарем, пропуск.")
                    continue

                cols = st.columns([3, 1]) # Изменил соотношение для кнопки, можно настроить
                with cols[0]:
                    cand_display_name = candidate_ml_data.get("Должность", f"Кандидат ID: {candidate_key}")
                    # Попытка взять ФИО, если оно уже есть в данных (маловероятно до раскрытия ПД, но на всякий случай)
                    if candidate_ml_data.get("ФИО"):
                        cand_display_name = f"{candidate_ml_data.get('ФИО')} ({cand_display_name})"

                    cand_score = candidate_ml_data.get("sim_score_second", 0)
                    if isinstance(cand_score, (float, int)) and not isinstance(cand_score, bool):
                        cand_score_perc = round(cand_score * 100 if cand_score <= 1.0 and cand_score !=0 else cand_score)
                    else:
                        cand_score_perc = "N/A"
                    st.markdown(f"**{cand_display_name}** (Скоринг: {cand_score_perc}%)")
                
                with cols[1]:
                    action_button_key = f"action_btn_{candidate_key}" # Упростил ключ кнопки
                    
                    if st.button("Раскрыть ПД и сохранить в Huntflow", key=action_button_key, help=f"Раскрыть ПД и создать {cand_display_name} в Huntflow"):
                        st.markdown(f"--- \n_Обработка: {cand_display_name}..._")
                        
                        with st.spinner(f"Работаем с {cand_display_name}..."):
                            pii_data_raw = None
                            access_token_ext = None # Access token для внешнего API (HH/Avito)
                            
                            candidate_link = candidate_ml_data.get("link")
                            source_type, resume_id_from_link = parse_resume_link(candidate_link)

                            if not source_type or not resume_id_from_link:
                                st.error(f"Не удалось извлечь источник/ID из ссылки: {candidate_link}")
                                logger.error(f"Ошибка парсинга ссылки для {cand_display_name}: {candidate_link}")
                                continue 
                            
                            logger.info(f"Для {cand_display_name}: источник='{source_type}', ID='{resume_id_from_link}'")

                            # 1. Получаем Access Token и затем PII
                            if source_type == "hh":
                                # access_token_ext = get_hh_oauth_token()
                                access_token_ext = st.secrets.get("HH_API_TOKEN")
                                if access_token_ext:
                                    pii_data_raw = get_hh_contacts_api(resume_id_from_link, access_token_ext)
                            elif source_type == "avito":
                                access_token_ext = get_avito_oauth_token()
                                if access_token_ext:
                                    pii_data_raw = get_avito_contacts_api(resume_id_from_link, access_token_ext)
                            else:
                                st.warning(f"Источник '{source_type}' не поддерживается для раскрытия ПД.")
                                logger.warning(f"Неподдерживаемый источник '{source_type}' для {cand_display_name}")
                                continue
                            
                            if not access_token_ext:
                                st.error(f"Не удалось получить Access Token для {source_type.upper()}.")
                                logger.error(f"Ошибка получения Access Token {source_type.upper()} для {cand_display_name}")
                                continue
                            
                            if not pii_data_raw:
                                st.error(f"Не удалось получить ПД для {cand_display_name} из {source_type.upper()}. Проверьте квоты или доступ к резюме.")
                                logger.error(f"Ошибка получения ПД {source_type.upper()} для {cand_display_name} (ID: {resume_id_from_link})")
                                continue
                            
                            st.success(f"ПД для {cand_display_name} из {source_type.upper()} получены.")
                            logger.info(f"ПД для {cand_display_name} ({source_type.upper()}) получены.")
                            pii_standardized = _extract_pii_details(pii_data_raw, source_type.upper()) # Передаем источник в верхнем регистре для _extract_pii_details
                            
                            # Опционально: отобразить полученные ПД для проверки
                            st.write("Извлеченные ПД:")
                            st.json(pii_standardized)
                            

                            hf_app_id, err_create = create_huntflow_applicant_api(pii_standardized, candidate_ml_data, resume_id_from_link, source_type.upper())
                            err_create = True
                            st.success(f"Создание кандидата в Huntflow захардкожено для тестов.")
                            logger.info(f"Создание кандидата в Huntflow: {hf_app_id} Захардкожено для тестов.")
                            if err_create:
                                st.error(f"Ошибка создания {cand_display_name} в Huntflow: {err_create}")
                                logger.error(f"Ошибка создания {cand_display_name} в Huntflow: {err_create}")
                                continue
                            st.success(f"Кандидат {cand_display_name} создан в Huntflow (ID: {hf_app_id}).")
                            logger.info(f"Кандидат {cand_display_name} создан в Huntflow (ID: {hf_app_id}).")

                            # 3. Заполняем анкету (упрощенно)
                            ok_q, err_q = fill_huntflow_questionary_api(hf_app_id, candidate_ml_data)
                            if err_q: 
                                st.warning(f"Ошибка при заполнении анкеты {cand_display_name} в Huntflow: {err_q}")
                                logger.warning(f"Ошибка анкеты {cand_display_name} в Huntflow: {err_q}")
                            else: 
                                st.info(f"Анкета для {cand_display_name} в Huntflow обработана.")
                                logger.info(f"Анкета для {cand_display_name} в Huntflow обработана.")

                            # 4. Привязываем к вакансии
                            current_hf_vacancy_id = st.session_state.get("selected_huntflow_vacancy_id")
                            if current_hf_vacancy_id:
                                # score_perc уже был рассчитан как cand_score_perc
                                ok_link, err_link = link_applicant_to_vacancy_api(hf_app_id, current_hf_vacancy_id, cand_score_perc if cand_score_perc != "N/A" else 0)
                                if err_link: 
                                    st.warning(f"Ошибка привязки {cand_display_name} к вакансии HF {current_hf_vacancy_id}: {err_link}")
                                    logger.warning(f"Ошибка привязки {cand_display_name} к вакансии HF {current_hf_vacancy_id}: {err_link}")
                                else: 
                                    st.info(f"{cand_display_name} привязан к вакансии HF {current_hf_vacancy_id}.")
                                    logger.info(f"{cand_display_name} привязан к вакансии HF {current_hf_vacancy_id}.")
                            else:
                                st.info(f"{cand_display_name} не будет привязан (вакансия Huntflow не выбрана).")
                                logger.info(f"{cand_display_name} не привязан к вакансии HF (не выбрана).")
                            
                            st.success(f"Кандидат {cand_display_name} полностью обработан для Huntflow!")
                            logger.info(f"Кандидат {cand_display_name} полностью обработан для Huntflow!")
                            # st.session_state[f"processed_hf_{candidate_key}"] = True # Пометить как обработанного
                            st.markdown("---")
                            # st.rerun() # Если нужно обновить UI немедленно (например, скрыть кнопку)