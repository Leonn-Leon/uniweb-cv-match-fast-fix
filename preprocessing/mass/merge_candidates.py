import pandas as pd
import json
import os
import glob
import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm
import logging
import concurrent.futures # Для параллелизации
import csv # Для работы с кэшем
import time # Для небольшой задержки при сохранении кэша
from datetime import date # Добавлено для работы с датами

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения (.env файл) ---
load_dotenv(override=True)
YANDEX_API_KEY = os.getenv("YANDEX_GEOCODER_API_KEY")
if not YANDEX_API_KEY:
    logger.warning("Ключ YANDEX_GEOCODER_API_KEY не найден в .env файле! Геокодирование будет невозможно или ограничено.")
    # exit()

# --- Константы ---
DATA_MASS_DIR = "data/raw/mass/" # Если папка лежит рядом со скриптом
GORNO_CSV_PATH = os.path.join(DATA_MASS_DIR, "Горнорабочий очистного забоя (ученик).csv")
BELAZ_CSV_PATH = os.path.join(DATA_MASS_DIR, "Водитель карьерного автосамосвала БелАЗ.csv")
OUTPUT_CSV_PATH = os.path.join("data/processed/mass", "candidates_hh.csv") # Выходной файл будет в data_mass/
GEOCODING_CACHE_CSV = os.path.join(DATA_MASS_DIR, "geocoding_cache.csv") # Кэш будет в preprocessing/data_mass/
MAX_GEOCODING_WORKERS = 10
LINK_COLUMN = "link"
ID_COLUMN = "ID"
DATE_COLUMN = "date"
JSON_DATE_SOURCE_COLUMN = "itemDate" # Колонка с датой в JSON

BASE_URL_AVITO = "https://www.avito.ru"
BASE_URL_HH = "https://hh.ru"
DEFAULT_BASE_URL = BASE_URL_AVITO

# Маппинг должностей для JSON файлов (русские названия)
JOB_TITLE_MAPPING = {
    "slesar_santehnik": "Слесарь-сантехник",
    "electromontajnik": "Электромонтажник",
    "injener_energetik": "Инженер-энергетик",
    "svarschik": "Сварщик",
    "voditel_samosvala": "Водитель самосвала",
     # Добавьте сюда остальные файлы из папки data_mass, если они есть
     # "имя_файла_без_расширения": "Русское Название Должности",
}

# Определяем целевые колонки и их порядок
FINAL_COLUMNS = [
    "link", "address", "Тип занятости", "Доступность", "Образование",
    "Компетенции", "График работы", "Переезд", "Опыт работы",
    "Категория прав", "Дата осмотра терапевтом", "ID", "Должность",
    "coords", "Описание", "date"
]

# Словарь для преобразования русских месяцев (в падеже из строки) в номер
MONTHS_RU_MAP = {
    "января": "01", "февраля": "02", "марта": "03", "апреля": "04",
    "мая": "05", "июня": "06", "июля": "07", "августа": "08",
    "сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"
}

# --- Функция парсинга даты из JSON ---
def parse_item_date(date_str: str) -> pd.Timestamp:
    """
    Пытается распарсить строку с датой из поля itemDate.
    Поддерживает форматы:
    1. ISO-подобный: "YYYY-MM-DDTHH:MM:SS.ffffff"
    2. Относительные: "· сегодня в HH:MM", "· вчера в HH:MM"
    3. Текстовые: "· DD MMMM в HH:MM", " · DD MMMM YYYY"
    Возвращает Timestamp или NaT (Not a Time) при ошибке.
    """
    if pd.isna(date_str) or not isinstance(date_str, str) or date_str.strip() == "":
        return pd.NaT

    try:
        # errors='coerce' вернет NaT, если формат не распознан, что нам и нужно
        # Не указываем format явно, чтобы pandas попробовал стандартные ISO-парсеры
        parsed_iso_date = pd.to_datetime(date_str, errors='coerce')
        if not pd.isna(parsed_iso_date):
            # Если успешно распознано, возвращаем (время будет отброшено при форматировании в YYYY-MM-DD позже)
            return parsed_iso_date
    except Exception:
        # Если прямой парсинг не удался, игнорируем ошибку и переходим к старой логике
        pass

    # --- Старая логика парсинга текстовых и относительных дат ---
    cleaned_str = date_str.lower().replace('·', '').strip()
    today = date.today()

    try:
        if "сегодня" in cleaned_str:
            return pd.Timestamp("2025-05-03") # <-- ИЗМЕНЕННАЯ СТРОКА
        if "вчера" in cleaned_str:
            return pd.Timestamp("2025-05-02") # <-- ИЗМЕНЕННАЯ СТРОКА

        parts = cleaned_str.split()
        if len(parts) < 2: # Недостаточно частей для даты
             raise ValueError("Недостаточно частей для извлечения даты")

        day = parts[0]
        month_ru = parts[1] # например, 'апреля'
        month_num = MONTHS_RU_MAP.get(month_ru)

        if not month_num:
            raise ValueError(f"Неизвестный месяц: '{month_ru}'")

        # Определение года
        year = None
        # Ищем 4 цифры в строке (может быть год)
        for part in parts:
            if part.isdigit() and len(part) == 4:
                year = part
                break
        # Если год не найден явно, считаем текущим
        if year is None:
            year = str(today.year)

        # Формируем дату в ISO формате (YYYY-MM-DD)
        # zfill(2) добавит ведущий ноль для дней < 10
        date_iso = f"{year}-{month_num}-{day.zfill(2)}"
        # Преобразуем в Timestamp, errors='coerce' вернет NaT при ошибке формата
        parsed_date = pd.to_datetime(date_iso, format="%Y-%m-%d", errors='coerce')
        if pd.isna(parsed_date):
             raise ValueError(f"Не удалось преобразовать {date_iso} в дату")
        return parsed_date

    except (IndexError, ValueError, TypeError) as e:
        logger.debug(f"Ошибка парсинга даты '{date_str}': {e}. Исходная очищенная: '{cleaned_str}'")
        return pd.NaT

# --- Класс Geocoder ---
class Geocoder:
    def __init__(self, api_key):
        self.api_key = api_key
        if not self.api_key: logger.warning("API ключ Яндекс Геокодера не предоставлен.")
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1.2), reraise=True)
    def get_coords(self, address_search: str):
        if not self.api_key: return None, str(address_search)
        address_str = str(address_search).strip()
        if not address_str or pd.isna(address_search): return None, str(address_search)
        base_url = "https://geocode-maps.yandex.ru/v1/"
        params = {"apikey": self.api_key, "geocode": address_str, "format": "json", "results": 1}
        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            feature_member = data.get("response",{}).get("GeoObjectCollection",{}).get("featureMember",[])
            if not feature_member: return None, address_str
            geo_object = feature_member[0].get("GeoObject", {})
            point_str = geo_object.get("Point", {}).get("pos")
            formatted_address = geo_object.get("metaDataProperty", {}).get("GeocoderMetaData", {}).get("Address", {}).get("formatted", address_str)
            if point_str:
                lon, lat = map(float, point_str.split())
                return (lat, lon), formatted_address
            else: return None, formatted_address
        except requests.exceptions.Timeout: logger.error(f"Таймаут запроса к геокодеру для '{address_str}'"); return None, address_str
        except requests.exceptions.RequestException as e: logger.error(f"Ошибка запроса к геокодеру для '{address_str}': {e}"); raise
        except Exception as e: logger.error(f"Ошибка обработки геокодинга для '{address_str}': {e}"); return None, address_str

# --- Функции для работы с кэшем ---
def load_geocoding_cache(filename: str) -> dict:
    cache = {}
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'address' not in reader.fieldnames: return {} # Нет нужной колонки
            for row in reader:
                addr_key = row.get('address')
                if not addr_key: continue
                try:
                    lat, lon = row.get('latitude'), row.get('longitude')
                    coords = (float(lat), float(lon)) if lat and lon and lat != 'None' and lat != '' and lon != 'None' and lon != '' else None # Усилена проверка на пустые строки
                    formatted = row.get('formatted_address', addr_key)
                    cache[addr_key] = (coords, formatted)
                except (ValueError, TypeError): cache[addr_key] = (None, row.get('formatted_address', addr_key)) # Если координаты не парсятся
        logger.info(f"Загружено {len(cache)} записей из кэша геокодирования '{filename}'")
    except FileNotFoundError: logger.info(f"Файл кэша '{filename}' не найден.")
    except Exception as e: logger.error(f"Ошибка загрузки кэша '{filename}': {e}.")
    return cache

def save_geocoding_cache(filename: str, cache: dict):
    try:
        temp_filename = filename + ".tmp"
        with open(temp_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['address', 'latitude', 'longitude', 'formatted_address']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for addr, (coords, fmt) in cache.items():
                 # Записываем пустую строку вместо None для совместимости при чтении
                writer.writerow({'address': addr, 'latitude': coords[0] if coords else '', 'longitude': coords[1] if coords else '', 'formatted_address': fmt})
        os.replace(temp_filename, filename)
        logger.info(f"Сохранено {len(cache)} записей в кэш геокодирования '{filename}'")
        time.sleep(0.1)
    except Exception as e:
        logger.error(f"Ошибка сохранения кэша в '{filename}': {e}")
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except OSError: pass

# --- Вспомогательная функция для очистки адреса ---
def clean_address(addr):
    if pd.isna(addr): return None
    try: return str(addr).strip().split("\n")[0].strip()
    except Exception: return str(addr).strip()

def build_full_link(row_link, is_hh):
            # Если ссылка уже полная, не трогаем (простая проверка на http)
            if row_link.startswith('http://') or row_link.startswith('https://'):
                return row_link
            # Если ссылка пустая, оставляем пустой
            if not row_link:
                return ''

            if is_hh:
                base = BASE_URL_HH
                row_link = row_link.replace("resumes", "resume")
            else:
                base = BASE_URL_AVITO

            # Убираем возможный лишний слэш в начале относительной ссылки
            if row_link.startswith('/'):
                return base + row_link
            else:
                return base + '/' + row_link

# --- Основная логика скрипта ---
if __name__ == "__main__":
    geocoder = Geocoder(YANDEX_API_KEY)
    geocoding_cache = load_geocoding_cache(GEOCODING_CACHE_CSV)
    newly_geocoded = {}

    # --- 1. Загрузка и объединение JSON файлов ---
    json_files = glob.glob(os.path.join(DATA_MASS_DIR, "*.json"))
    logger.info(f"Найдено {len(json_files)} JSON файлов: {json_files}")
    all_json_dfs = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = json.load(f) # Загружаем содержимое файла

            # Определяем, какая структура у файла: старая (список) или новая (словарь с "candidates")
            candidate_list_raw = []
            if isinstance(file_content, list):
                # Старый формат: файл содержит список словарей
                candidate_list_raw = file_content
                logger.debug(f"Файл {file_path} определен как старый формат (список словарей).")
            elif isinstance(file_content, dict) and "candidates" in file_content and isinstance(file_content["candidates"], list):
                # Новый формат: файл содержит словарь с ключом "candidates"
                candidate_list_raw = file_content["candidates"]
                logger.debug(f"Файл {file_path} определен как новый формат (словарь с 'candidates').")
            else:
                logger.warning(f"Пропуск файла {file_path}: неизвестный формат JSON или отсутствует ключ 'candidates'.")
                continue

            if not candidate_list_raw: # Если список кандидатов пуст
                logger.info(f"Файл {file_path} не содержит кандидатов или список 'candidates' пуст. Пропуск.")
                continue

            # В df_init теперь будут словари кандидатов (старый формат)
            # или словари из списка "candidates" (новый формат)
            df_init = pd.DataFrame(candidate_list_raw)

            file_name = os.path.basename(file_path)
            job_key = os.path.splitext(file_name)[0]
            # fallback_job_title определяется как и раньше, на случай если title не будет найден
            fallback_job_title = JOB_TITLE_MAPPING.get(job_key) # Убрал None по умолчанию, т.к. ниже проверка if not fallback_job_title
            if not fallback_job_title: # Если в маппинге нет, используем имя файла
                fallback_job_title = job_key.replace('_', ' ').capitalize()
                logger.warning(f"Не найдено точное соответствие для файла '{file_name}' в JOB_TITLE_MAPPING. Fallback должность: '{fallback_job_title}' (или будет взята из поля 'title' в resume_schema).")


            processed_rows = []
            for index, row_candidate_outer in df_init.iterrows():
                # `row_candidate_outer` - это один словарь кандидата из списка
                # (либо из старого формата, либо из `file_content["candidates"]`)

                # Инициализируем flat_row с полями верхнего уровня кандидата
                flat_row = row_candidate_outer.to_dict()

                # Пытаемся извлечь и распарсить resume_schema, если она есть
                resume_schema_str = flat_row.pop("resume_schema", None) # Извлекаем и удаляем, чтобы не мешала дальше

                if resume_schema_str and isinstance(resume_schema_str, str):
                    try:
                        resume_data = json.loads(resume_schema_str)
                        flat_row.update(resume_data)
                    except json.JSONDecodeError as e_schema:
                        logger.warning(f"Ошибка декодирования resume_schema в файле {file_path}, строка {index}: {e_schema}. Данные из resume_schema не будут использованы.")
                elif "resume_schema" in row_candidate_outer.to_dict(): # Если resume_schema было, но не строка
                     logger.warning(f"Поле resume_schema в файле {file_path}, строка {index} не является строкой. Данные из resume_schema не будут использованы.")

                for key_nested in ["mainParams", "additionalParams", "techInfo"]:
                    if key_nested in flat_row and isinstance(flat_row[key_nested], dict):
                        for sub_key, sub_value in flat_row[key_nested].items():
                            if sub_key not in flat_row:
                                flat_row[sub_key] = sub_value
                        del flat_row[key_nested] # Удаляем сам вложенный ключ

                fields_potentially_list = ["Опыт работы", "Категория прав", "Учебные заведения", "Гражданство"]
                for field_name in fields_potentially_list:
                    if field_name in flat_row and isinstance(flat_row[field_name], list):
                        try:
                            string_items = [str(item).strip() for item in flat_row[field_name] if item is not None and str(item).strip() != ""]
                            # Оставляем "Нет данных", если список пуст после фильтрации
                            flat_row[field_name] = "\n".join(string_items) if string_items else "Нет данных"
                        except Exception as e_list:
                            logger.warning(f"Ошибка обработки списка '{field_name}' в {file_path}, строка {index}: {e_list}. Установлено 'Нет данных'.")
                            flat_row[field_name] = "Нет данных"

                # Определение должности
                # Приоритет у 'title' из resume_schema (он уже в flat_row, если был)
                record_title = flat_row.get("title")
                if record_title and isinstance(record_title, str) and record_title.strip():
                    final_job_title = record_title.strip()
                else:
                    # Если title нет в resume_schema или он пустой, используем fallback
                    final_job_title = fallback_job_title
                flat_row["Должность"] = final_job_title

                # Обработка описания
                # 'description' должно прийти из resume_schema (уже в flat_row, если было)
                desc_col_target = 'Описание' # Целевое имя колонки
                if 'description' in flat_row:
                    # Если 'description' существует в flat_row, переименовываем/переносим его в 'Описание'
                    # Если 'Описание' уже существует (маловероятно, но возможно), оно будет перезаписано
                    flat_row[desc_col_target] = flat_row.pop('description') # pop извлекает и удаляет
                # Если 'description' не было, поле 'Описание' останется как есть (или будет None)

                processed_rows.append(flat_row)

            df_processed = pd.DataFrame(processed_rows)
            all_json_dfs.append(df_processed)
            logger.info(f"Обработан {file_path}, строк: {len(df_processed)}")

        except json.JSONDecodeError: logger.error(f"Ошибка декодирования JSON на уровне файла: {file_path}")
        except Exception as e: logger.error(f"Не удалось обработать файл {file_path}: {e}")
    if not all_json_dfs: logger.error("Не обработано ни одного JSON файла. Выход."); exit()

    df_json_combined = pd.concat(all_json_dfs, ignore_index=True, sort=False)
    logger.info(f"Объединенные JSON данные: строк={len(df_json_combined)}, колонок={len(df_json_combined.columns)}")

    # --- 1.1 Переименование itemId -> ID для JSON ---
    if 'itemId' in df_json_combined.columns:
        df_json_combined.rename(columns={'itemId': ID_COLUMN}, inplace=True)
        logger.info(f"Переименована колонка 'itemId' в '{ID_COLUMN}'.")
    else:
        logger.warning(f"Колонка 'itemId' не найдена в JSON для переименования в '{ID_COLUMN}'.")

    # df_json_combined[ID_COLUMN] = df_json_combined[LINK_COLUMN].apply(lambda x: "№ "+str(x[-10:]))

    # --- 1.2 Парсинг дат из JSON ---
    if JSON_DATE_SOURCE_COLUMN in df_json_combined.columns:
        parsed_dates = df_json_combined[JSON_DATE_SOURCE_COLUMN].apply(parse_item_date)
        df_json_combined[DATE_COLUMN] = parsed_dates.dt.strftime('%Y-%m-%d') # NaT станут NaN/пустотами
        total_dates = df_json_combined[JSON_DATE_SOURCE_COLUMN].notna().sum()
        success_dates = df_json_combined[DATE_COLUMN].notna().sum()
        logger.info(f"Парсинг дат из '{JSON_DATE_SOURCE_COLUMN}': Успешно {success_dates}/{total_dates}")
    else:
        logger.warning(f"Колонка '{JSON_DATE_SOURCE_COLUMN}' не найдена в JSON. Колонка '{DATE_COLUMN}' будет пустой для JSON.")
        df_json_combined[DATE_COLUMN] = None

    # --- 2. Загрузка и подготовка CSV файлов ---  (Перенесено ДО объединения)
    all_csv_dfs = []
    for csv_path in [GORNO_CSV_PATH, BELAZ_CSV_PATH]:
        try:
            df_csv = pd.read_csv(csv_path)
            logger.info(f"Загружен {csv_path}, строк: {len(df_csv)}")
            # Проверяем наличие необходимых колонок, добавляем с None если отсутствуют
            if "Должность" not in df_csv.columns:
                logger.warning(f"В {csv_path} отсутствует 'Должность'. Будет None.")
                df_csv["Должность"] = None
            # !!! Удалена строка добавления coords=None !!!
            if DATE_COLUMN not in df_csv.columns:
                logger.warning(f"В {csv_path} отсутствует '{DATE_COLUMN}'. Будет None.")
                df_csv[DATE_COLUMN] = None
            # Опциональный парсинг даты для CSV, если формат отличается
            # if DATE_COLUMN in df_csv.columns:
            #      df_csv[DATE_COLUMN] = pd.to_datetime(df_csv[DATE_COLUMN], errors='coerce').dt.strftime('%Y-%m-%d')
            all_csv_dfs.append(df_csv)
        except FileNotFoundError: logger.warning(f"Файл не найден: {csv_path}. Пропуск.")
        except Exception as e: logger.error(f"Ошибка загрузки {csv_path}: {e}. Пропуск.")

    # --- 3. Объединение всех данных --- (Перенесено ДО геокодирования)
    all_dfs_to_combine = [df_json_combined] + all_csv_dfs

    df_combined = pd.concat(all_dfs_to_combine, ignore_index=True, sort=False)
    logger.info(f"Все данные объединены: строк={len(df_combined)}, колонок={len(df_combined.columns)}")

    # --- 4. Очистка адресов и Параллельное геокодирование --- (Перенесено ПОСЛЕ объединения)
    # Теперь применяется к df_combined
    if "address" in df_combined.columns:
        logger.info("Очистка адресов и геокодирование для ВСЕХ данных...")
        # Очищаем адреса для всего датафрейма
        df_combined['address'] = df_combined['address'].apply(clean_address)

        unique_addresses = df_combined['address'].dropna().unique()
        addresses_to_geocode = [addr for addr in unique_addresses if addr not in geocoding_cache or geocoding_cache[addr][0] is None]
        cached_ok_count = len(unique_addresses) - len(addresses_to_geocode)
        logger.info(f"Уникальных адресов: {len(unique_addresses)}. Успешно в кэше: {cached_ok_count}. К обработке: {len(addresses_to_geocode)}")

        if addresses_to_geocode and geocoder.api_key:
            logger.info(f"Запуск геокодирования ({MAX_GEOCODING_WORKERS} потоков)...")
            failed_addresses = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_GEOCODING_WORKERS) as executor:
                future_to_address = {executor.submit(geocoder.get_coords, addr): addr for addr in addresses_to_geocode}
                for future in tqdm(concurrent.futures.as_completed(future_to_address), total=len(addresses_to_geocode), desc="Геокодирование"):
                    address = future_to_address[future]
                    try:
                        coords, formatted_address = future.result()
                        newly_geocoded[address] = (coords, formatted_address)
                        if coords is None: failed_addresses += 1
                    except Exception as exc:
                        logger.error(f'Ошибка геокодирования "{address}": {exc}')
                        newly_geocoded[address] = (None, address)
                        failed_addresses += 1
            logger.info(f"Геокодирование завершено. Результатов: {len(newly_geocoded)}. Ошибок координат: {failed_addresses}.")
            if newly_geocoded: # Сохраняем кэш только если были новые попытки
                logger.info("Обновление кэша...")
                geocoding_cache.update(newly_geocoded)
                save_geocoding_cache(GEOCODING_CACHE_CSV, geocoding_cache)
        elif not geocoder.api_key and addresses_to_geocode:
             logger.error("Пропуск геокодирования: отсутствует API ключ.")
        else: logger.info("Нет новых адресов для геокодирования.")

        # Применение координат из ПОЛНОГО кэша к df_combined
        logger.info("Применение координат к DataFrame...")
        def get_coords_from_cache(addr, cache): return cache.get(addr, (None, None))[0]
        # Создаем колонку coords для ВСЕХ данных
        df_combined['coords'] = df_combined['address'].map(lambda x: get_coords_from_cache(x, geocoding_cache))
        logger.info(f"Строк без координат в объединенных данных: {df_combined['coords'].isna().sum()}/{len(df_combined)}")
    else:
        logger.warning("Колонка 'address' не найдена в объединенных данных. Геокодирование пропущено.")
        df_combined['coords'] = None # Добавляем пустую колонку для совместимости

    # --- 5. Удаление дубликатов ---
    rows_before_dedup = len(df_combined)
    if LINK_COLUMN in df_combined.columns:
        logger.info(f"Удаление дубликатов по '{LINK_COLUMN}'...")
        df_combined.drop_duplicates(subset=[LINK_COLUMN], keep='first', inplace=True, ignore_index=True)
        logger.info(f"Строк после удаления дубликатов по '{LINK_COLUMN}': {len(df_combined)}")
    
    is_hh_source = pd.Series([False] * len(df_combined), index=df_combined.index)
    if 'source' in df_combined.columns:
        is_hh_source = df_combined['source'].astype(str).str.lower() == 'headhunter'
    else:
        logger.warning("Колонка 'source' не найдена. Все ссылки будут обработаны с DEFAULT_BASE_URL.")
    
    df_combined[LINK_COLUMN] = df_combined.apply(
        lambda row: build_full_link(row[LINK_COLUMN], is_hh_source[row.name]),
        axis=1
    )
    logger.info(f"Модификация колонки '{LINK_COLUMN}' завершена.")

    if ID_COLUMN in df_combined.columns:
         logger.info(f"Дополнительное удаление дубликатов по '{ID_COLUMN}'...")
         id_na = df_combined[ID_COLUMN].isna() | (df_combined[ID_COLUMN] == '') | (df_combined[ID_COLUMN] == 'ID отсутствует')
         df_combined_valid_id = df_combined[~id_na]
         df_combined_invalid_id = df_combined[id_na]
         df_combined_valid_id.drop_duplicates(subset=[ID_COLUMN], keep='first', inplace=True)
         df_combined = pd.concat([df_combined_valid_id, df_combined_invalid_id], ignore_index=True)
         logger.info(f"Строк после дополнительного удаления дубликатов по '{ID_COLUMN}': {len(df_combined)}")

    rows_after_dedup = len(df_combined)
    if rows_before_dedup > rows_after_dedup:
        logger.info(f"Всего удалено дубликатов: {rows_before_dedup - rows_after_dedup}")

    # --- 6. Формирование итогового DataFrame ---
    df_final = pd.DataFrame()
    logger.info("Формирование итогового DataFrame...")
    missing_cols = []
    for col in FINAL_COLUMNS:
        if col in df_combined.columns: df_final[col] = df_combined[col]
        else: df_final[col] = None; missing_cols.append(col)
    if missing_cols: logger.warning(f"Целевые колонки добавлены с None: {missing_cols}")

    df_final = df_final[FINAL_COLUMNS] # Гарантируем порядок

    # --- 7. Финальная обработка и очистка ---
    # Удаление строк с невалидными датами
    if DATE_COLUMN in df_final.columns:
        rows_before_dropna = len(df_final)
        df_final = df_final[df_final[DATE_COLUMN].notna()].copy()
        rows_after_dropna = len(df_final)
        if rows_before_dropna > rows_after_dropna:
             logger.info(f"Удалено {rows_before_dropna - rows_after_dropna} строк с отсутствующими датами.")
    else:
        logger.warning(f"Колонка '{DATE_COLUMN}' не найдена, пропуск удаления строк по дате.")

    # Замена пропусков ID
    if ID_COLUMN in df_final.columns:
         condition = df_final[ID_COLUMN].isna() | (df_final[ID_COLUMN] == '')
         replaced_count = condition.sum()
         df_final.loc[condition, ID_COLUMN] = "ID отсутствует"
         logger.info(f"Заменено {replaced_count} пропусков в '{ID_COLUMN}'.")

    # Общая замена остальных пропусков
    logger.info("Замена остальных пропусков (NaN/NaT/None) на 'Нет данных'...")
    df_final = df_final.fillna("Нет данных")

    # Замена пустых строк на "Нет данных"
    df_final = df_final.replace('', 'Нет данных')

    # Приведение к строковому типу
    cols_to_stringify = ['Описание', 'Опыт работы']
    for col in cols_to_stringify:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(str)
            logger.info(f"Колонка '{col}' преобразована в строку.")

    logger.info(f"Итоговый DataFrame: строк={len(df_final)}, колонок={len(df_final.columns)}")
    logger.info(f"Колонки: {df_final.columns.tolist()}")
    print("\nПредпросмотр итогового DataFrame (первые 5 строк):")
    # print(df_final.head().to_markdown(index=False)) # Вывод в формате Markdown

    # --- 8. Сохранение результата ---
    try:
        # Создаем директорию для выходного файла, если ее нет
        output_dir = os.path.dirname(OUTPUT_CSV_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Создана директория для вывода: {output_dir}")

        df_final.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        logger.info(f"Итоговый DataFrame успешно сохранен в {OUTPUT_CSV_PATH}")
    except Exception as e:
        logger.error(f"Не удалось сохранить итоговый DataFrame в {OUTPUT_CSV_PATH}: {e}")