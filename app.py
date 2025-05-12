# app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from copy import deepcopy
from loguru import logger
import datetime # Добавлен импорт

# Импорт ваших модулей
from src.utils.utils import Mode, load_data, load_model, df2dict
from src.ui import common_ui, mass_ui, prof_ui

# --- Инициализация Session State ---
if "computed" not in st.session_state:
    st.session_state["computed"] = False
if "vahta_mode" not in st.session_state:
    st.session_state["vahta_mode"] = False
if "distance_option" not in st.session_state:
    st.session_state["distance_option"] = "Нет ограничений"
if "original_address_weight" not in st.session_state: # Для коллбэка
    st.session_state.original_address_weight = None

st.title("Подбор кандидатов 💼") # Изменил для общности, можно вернуть

current_mode = Mode.MASS # Пока жестко задаем массовый подбор

# --- Загрузка данных и модели в зависимости от режима ---
if current_mode == Mode.PROF:
    # vacancy_df = load_data(path="./data/vacancies.csv") # Если нужен список вакансий
    selector, config = load_model(config_path="./config/config.yaml")
else: # Mode.MASS
    # vacancy_df = load_data(path="./data_mass/vacancies.csv") # Если нужен
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
                if current_weight == 0.0: # Восстанавливаем только если был обнулен
                    df.loc[address_index, 'Вес'] = st.session_state.original_address_weight
                    st.toast(f"Вес 'Адрес' восстановлен: {st.session_state.original_address_weight}", icon="👍")
        st.session_state["df_weights"] = df
    except Exception as e:
        st.error(f"Ошибка при обновлении веса 'Адрес': {e}")

# --- Отображение UI ---
common_ui.display_sidebar(config, update_address_weight_callback)

vacancy_input_data = None
if current_mode == Mode.MASS:
    st.header("Массовый подбор кандидатов")
    vacancy_input_data = mass_ui.display_mass_input_form()
elif current_mode == Mode.PROF:
    st.header("Профессиональный подбор кандидатов")
    vacancy_input_data = prof_ui.display_prof_input_form() # Пока вернет None

# --- Логика подбора по нажатию кнопки ---
if st.button("Подобрать", type="primary"):
    if not vacancy_input_data:
        st.warning("Режим подбора не вернул данные для вакансии. Возможно, он еще не реализован.")
    elif (
        vacancy_input_data.get("Должность")
        and vacancy_input_data.get("required")
        and vacancy_input_data.get("location")
    ):
        st.session_state["computed"] = False # Сбрасываем перед новым подбором
        
        # Получаем актуальные значения из session_state, установленные в сайдбаре
        selected_date_threshold = st.session_state.selected_date_threshold
        threshold_from_slider = st.session_state.current_threshold # Это и есть наш score_threshold_stage_2
        distance_option_val = st.session_state.distance_option
        is_vahta_val = st.session_state.vahta_mode
        
        # Загрузка данных кандидатов (специфично для режима)
        if current_mode == Mode.PROF:
            st.info("Логика проф. подбора еще не полностью реализована.")
            st.stop() # Пока остановим здесь для проф режима
        else: # Mode.MASS
            df_cv = load_data(f"data/processed/mass/candidates_hh.csv")
            if df_cv.empty:
                st.error("Не удалось загрузить данные кандидатов.")
                st.stop()
            df_cv = df_cv.rename(columns={"address": "Адрес"})

        with st.status("Подготовка вакансии..."):
            vacancy_processed = selector.preprocess_vacancy(deepcopy(vacancy_input_data)) # Используем deepcopy
        
        with st.status("Подбор кандидатов..."):
            # --- Логика кэширования (можно вынести в функцию) ---
            use_cache = not (not Path("./tmp_cvs.csv").exists() or config["general"]["mode"] == "prod")
            
            if not use_cache:
                st.write(f"Первая фаза: анализ {df_cv.shape[0]} кандидатов...")
                
                max_distance_filter = None
                if distance_option_val != "Нет ограничений":
                    max_distance_filter = float(distance_option_val)

                # Важно: используем актуальные веса из st.session_state["df_weights"] для 2го этапа
                # Для 1го этапа веса могут быть из config или модифицированы "Вахтой"
                current_first_stage_weights = deepcopy(config["model"]["stage_1"]["weights"]) # Копируем, чтобы не менять конфиг
                if is_vahta_val:
                    # Предполагаем, что вес "Адрес" на 1-м этапе - второй элемент (индекс 1)
                    # Это нужно будет адаптировать, если структура весов другая
                    if len(current_first_stage_weights) > 1: 
                        current_first_stage_weights[1] = 0.0
                    else:
                        logger.warning("Не удалось обнулить вес 'Адрес' для 1-го этапа: недостаточно весов в конфиге.")


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
                    df_weights=st.session_state["df_weights"], # Используем актуальные веса из сайдбара
                    score_threshold_stage_2=threshold_from_slider # Используем актуальный порог из сайдбара
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

            # Сохраняем результаты в session_state для отображения
            st.session_state.data_cv_dict = df2dict(df_ranked_2nd) # Конвертируем в dict для UI
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

# --- Отображение результатов (после нажатия кнопки и успешного вычисления) ---
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
                    config, # Передаем весь config, т.к. display_mass_results его использует
                    nan_mask_to_display
                )
            else:
                st.error("Ошибка в структуре конфигурационного файла для отображения результатов.")

        elif current_mode == Mode.PROF:
            # Для проф. режима может потребоваться другая маска или ее обработка
            # nan_mask_prof = np.delete(nan_mask_to_display, [1, 2, 5]) # Пример из вашего кода
            prof_ui.display_prof_results(
                data_cv_to_display, 
                vacancy_prep_to_display, 
                config, 
                nan_mask_to_display # или nan_mask_prof
            )