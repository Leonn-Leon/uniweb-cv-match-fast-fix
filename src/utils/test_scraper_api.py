import streamlit as st
import requests
import time
import json
import re
from loguru import logger

# --- КОНФИГУРАЦИЯ ---
# Базовые URL из секретов
HUNTFLOW_BASE_URL = st.secrets.get("HUNTFLOW_BASE_URL")
SCRAPER_API_BASE_URL = st.secrets.get("SCRAPER_API_BASE_URL")
SCRAPER_API_TOKEN = st.secrets.get("SCRAPER_API_TOKEN") # Будет None, если не задан

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С HUNTFLOW (твой код) ---

def get_huntflow_proxies():
    proxy_user = st.secrets.get("HUNTFLOW_PROXY_USER")
    proxy_pass = st.secrets.get("HUNTFLOW_PROXY_PASS")
    proxy_host = st.secrets.get("HUNTFLOW_PROXY_HOST")
    proxy_port = st.secrets.get("HUNTFLOW_PROXY_PORT")

    if all([proxy_user, proxy_pass, proxy_host, proxy_port]):
        proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        return {"http": proxy_url, "https": proxy_url}
    return None

def fetch_huntflow_vacancies_api():
    """Загружает активные вакансии из Huntflow."""
    proxies = get_huntflow_proxies()
    try:
        hf_token = st.secrets.get("HUNTFLOW_API_TOKEN")
        hf_account_id = st.secrets.get("HUNTFLOW_ACCOUNT_ID", "2")
        if not hf_token:
            st.error("Токен API Huntflow не найден в секретах.")
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
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при загрузке вакансий из Huntflow: {e}")
    except Exception as e:
        st.error(f"Неожиданная ошибка (вакансии Huntflow): {e}")
    return [], {}

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С RESUME SCRAPER API ---

def get_scraper_api_headers():
    """Формирует заголовки для запросов к API скрапера."""
    headers = {"Content-Type": "application/json"}
    if SCRAPER_API_TOKEN:
        headers["Authorization"] = f"Bearer {SCRAPER_API_TOKEN}"
    return headers

def build_search_params(vacancy_details, aggregator_type):
    """Шаг 1: Конвертирует вакансию в параметры поиска."""
    st.info(f"1. Конвертируем вакансию в параметры для {aggregator_type}...")
    
    url = f"{SCRAPER_API_BASE_URL}/parser/build-search-urls?aggregator_type={aggregator_type}"
    
    # Собираем тело запроса из данных вакансии
    # Убедись, что ключи совпадают с ожидаемыми в API
    payload = {
        "position": vacancy_details.get("position"),    
        "money": re.sub(r'\D', '', vacancy_details.get("money")),
        "region_rusal": vacancy_details.get("company_region"), # Пример: может быть другое поле
        "experience_position": vacancy_details.get("experience"), # Пример
        "education": vacancy_details.get("education") # Пример
    }
    # Убираем пустые значения
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        response = requests.post(url, headers=get_scraper_api_headers(), json=payload)
        response.raise_for_status()
        result = response.json()
        st.success("Параметры успешно созданы.")
        st.json(result.get("search_params"))
        return result.get("search_params")
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при создании параметров поиска: {e}")
        st.json(e.response.json())
        return None

def launch_pipeline(search_params, aggregator_type):
    """Шаг 2: Запускает полный pipeline."""
    st.info(f"2. Запускаем pipeline для {aggregator_type}...")
    
    pipeline_name = aggregator_type.replace('_', '-') # headhunter_api -> headhunter-api
    url = f"{SCRAPER_API_BASE_URL}/pipeline/{pipeline_name}/launch"
    
    payload = {
        "search_params": search_params,
        "run_aggregator": True,
        "run_scraper": True,
        "run_parser": True
    }
    
    try:
        response = requests.post(url, headers=get_scraper_api_headers(), json=payload)
        response.raise_for_status()
        result = response.json()
        task_id = result.get("id")
        st.success(f"Pipeline запущен! Task ID: {task_id}")
        return task_id
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка при запуске pipeline: {e}")
        st.json(e.response.json())
        return None

def track_task_progress(task_id):
    """Шаг 3 и 4: Отслеживает прогресс и получает результат."""
    st.info(f"3. Отслеживаем выполнение задачи {task_id}...")
    
    url = f"{SCRAPER_API_BASE_URL}/tasks/{task_id}"
    progress_bar = st.progress(0, text="Ожидание запуска...")
    status_text = st.empty()

    while True:
        try:
            response = requests.get(url, headers=get_scraper_api_headers())
            response.raise_for_status()
            task_info = response.json()
            
            status = task_info.get("status")
            progress = task_info.get("progress", 0)
            message = task_info.get("message", "")

            if progress is None:
                progress = 0
            
            progress_bar.progress(progress, text=f"{status}: {message}")
            status_text.json(task_info) # Показываем полный JSON ответа для отладки
            logger.info("Ответ: "+str(task_info))
            

            if status in ["success", "failed", "cancelled"]:
                st.info("4. Задача завершена!")
                if status == "success":
                    st.success("Задача выполнена успешно!")
                else:
                    st.error(f"Задача завершилась со статусом: {status}")
                
                st.subheader("Итоговый результат:")
                st.json(task_info.get("result", {}))
                break

            time.sleep(5) # Пауза 5 секунд между проверками
            
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при проверке статуса задачи: {e}")
            break

# --- ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(layout="wide")
st.title("Тестирование API для поиска резюме")

if not SCRAPER_API_BASE_URL:
    st.error("Необходимо задать `SCRAPER_API_BASE_URL` в .streamlit/secrets.toml")
else:
    st.info(f"API для тестирования: `{SCRAPER_API_BASE_URL}`")

    # Загружаем вакансии
    vacancies_list, vacancies_map = fetch_huntflow_vacancies_api()

    if not vacancies_list:
        st.warning("Не удалось загрузить вакансии из Huntflow.")
    else:
        # Выбор вакансии
        selected_vacancy_option = st.selectbox(
            "Выберите вакансию из Huntflow для теста:",
            options=vacancies_list,
            format_func=lambda x: x[0] # Показываем только название вакансии
        )
        
        selected_vacancy_id = selected_vacancy_option[1]
        selected_vacancy_details = vacancies_map.get(selected_vacancy_id)

        st.subheader("Детали выбранной вакансии (JSON из Huntflow)")
        st.json(selected_vacancy_details)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.header("Поиск на HeadHunter")
            if st.button("Запустить поиск на HH.ru", key="hh"):
                # Полный цикл для HH
                with st.spinner("Выполняется полный цикл для HH..."):
                    params = build_search_params(selected_vacancy_details, "headhunter_api")
                    if params:
                        task_id = launch_pipeline(params, "headhunter_api")
                        if task_id:
                            track_task_progress(task_id)

        with col2:
            st.header("Поиск на Avito")
            if st.button("Запустить поиск на Avito", key="avito"):
                # Полный цикл для Avito
                with st.spinner("Выполняется полный цикл для Avito..."):
                    params = build_search_params(selected_vacancy_details, "avito_api")
                    if params:
                        task_id = launch_pipeline(params, "avito_api")
                        if task_id:
                            track_task_progress(task_id)