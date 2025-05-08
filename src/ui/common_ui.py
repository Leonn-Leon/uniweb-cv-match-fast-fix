# common_ui.py
import streamlit as st
import pandas as pd
import datetime
from copy import deepcopy
from loguru import logger # Если используете logger

from utils.hardcode_data import map_names # Импортируем map_names

def display_sidebar(config, update_address_weight_callback_func):
    """Отображает сайдбар с настройками."""
    with st.sidebar:
        st.header("Информация ℹ️")
        st.write("Веса компонент скоринга.")

        st.checkbox(
            "Вахта",
            key="vahta_mode",
            on_change=update_address_weight_callback_func
        )

        if "df_weights" not in st.session_state: # Инициализация здесь, если еще нет
            features = deepcopy(config["model"]["stage_2"]["ranking_features"])
            info_dict = {"Компонента": [], "Вес": []}
            for feature, value in zip(features, config["model"]["stage_2"]["weights"]):
                info_dict["Компонента"].append(
                    feature if feature not in map_names else map_names[feature]
                )
                info_dict["Вес"].append(round(value, 2))
            st.session_state["df_weights"] = pd.DataFrame(info_dict)

        st.session_state["df_weights"] = st.data_editor(
            st.session_state["df_weights"],
            disabled=["Компонента"],
            hide_index=True,
            width=300,
        )
        st.divider()

        st.header("Настройки подбора")

        default_date_str = config["model"]["stage_1"]["date_threshold"]
        try:
            default_date = datetime.date.fromisoformat(default_date_str)
        except ValueError:
            logger.warning(f"Invalid date format in config for date_threshold: '{default_date_str}'. Using 2025-02-01.")
            default_date = datetime.date(2025, 2, 1)

        selected_date_threshold = st.date_input(
            "Рассматривать резюме не старше:",
            value=default_date,
            min_value=datetime.date(2015, 1, 1),
            max_value=datetime.date.today(),
            help="Резюме, опубликованные или обновленные до этой даты, будут отфильтрованы."
        )
        # Сохраняем в session_state, чтобы app.py мог это прочитать
        st.session_state.selected_date_threshold = selected_date_threshold


        threshold = st.slider(
            "Минимальный процент соответствия",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('current_threshold', 0.5), # Используем значение из session_state или дефолт
            step=0.05,
            help="Кандидаты с процентом соответствия ниже этого значения не будут показаны"
        )
        # Обновляем session_state при изменении слайдера
        st.session_state.current_threshold = threshold


        st.selectbox(
            label="Ограничение по расстоянию (км):",
            options=["Нет ограничений", "100", "500", "1000", "5000"],
            key="distance_option",
            help="Фильтрует кандидатов на 1-м этапе по расстоянию до вакансии."
        )