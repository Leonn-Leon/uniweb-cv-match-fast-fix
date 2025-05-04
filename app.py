import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.hardcode_data import candidate_names, map_names
from utils.utils import *
import datetime
from loguru import logger

if "computed" not in st.session_state:
    st.session_state["computed"] = False

if "vahta_mode" not in st.session_state:
    st.session_state["vahta_mode"] = False # По умолчанию галочка

if "distance_option" not in st.session_state:
    st.session_state["distance_option"] = "Нет ограничений"

st.title("Массовый подбор кандидатов 💼")

# mode = st.select_slider(
#     "Выберете тип подбора",
#     options=[
#         str(Mode.MASS),
#         str(Mode.PROF),
#     ],
# )

mode = Mode.MASS

if mode == str(Mode.PROF):
    vacancy_df = load_data(path="./data/vacancies.csv")
    selector, config = load_model(config_path="./config/config.yaml")
else:
    # vacancy_df = load_data(path="./data_mass/vacancies.csv")
    selector, config = load_model(config_path="./config/config_mass.yaml")

# vacancies = vacancy_df["Должность"].to_list()

if "df_weights" not in st.session_state:
    features = deepcopy(config["model"]["stage_2"]["ranking_features"])
    info_dict = {"Компонента": [], "Вес": []}
    for feature, value in zip(features, config["model"]["stage_2"]["weights"]):
        info_dict["Компонента"].append(
            feature if feature not in map_names else map_names[feature]
        )
        info_dict["Вес"].append(round(value, 2))

    df_weights = pd.DataFrame(info_dict)
    st.session_state["df_weights"] = df_weights
if 'current_threshold' not in st.session_state:
    st.session_state['current_threshold'] = config["model"]["stage_2"]["score_threshold"]


def update_address_weight_callback():
    """Обновляет вес 'Адрес' в df_weights на основе галочки 'Вахта'."""
    try:
        df = st.session_state["df_weights"]
        is_vahta = st.session_state["vahta_mode"] # Текущее состояние галочки

        # Находим индекс строки 'Адрес'
        address_index_list = df.index[df['Компонента'] == 'Адрес'].tolist()

        if not address_index_list:
            # Если строки 'Адрес' нет, ничего не делаем
            return

        address_index = address_index_list[0]
        current_weight = df.loc[address_index, 'Вес']

        if is_vahta:
            # Галочка ВКЛЮЧЕНА
            # Если вес еще не 0, сохраняем его как оригинальный и ставим 0
            if current_weight != 0.0:
                st.session_state.original_address_weight = current_weight
                df.loc[address_index, 'Вес'] = 0.0
                st.toast("Вес 'Адрес' установлен в 0 (Вахта).", icon="⚠️") # Уведомление

        else:
            # Галочка ВЫКЛЮЧЕНА
            # Восстанавливаем оригинальный вес, если он был сохранен
            if st.session_state.original_address_weight is not None:
                # Восстанавливаем только если текущий вес 0 (чтобы не перезаписать ручные изменения)
                if current_weight == 0.0:
                     df.loc[address_index, 'Вес'] = st.session_state.original_address_weight
                     st.toast(f"Вес 'Адрес' восстановлен: {st.session_state.original_address_weight}", icon="👍")

            # Если оригинальный вес не сохранен (маловероятно), ничего не делаем

        # Обновляем DataFrame в session_state, чтобы data_editor перерисовался
        st.session_state["df_weights"] = df

    except Exception as e:
        st.error(f"Ошибка при обновлении веса 'Адрес': {e}")

with st.sidebar:
    st.header("Информация ℹ️")
    st.write("Веса компонент скоринга.")

    st.checkbox(
        "Вахта",
        key="vahta_mode",
        on_change=update_address_weight_callback 
        )

    st.session_state["df_weights"] = st.data_editor(
        st.session_state["df_weights"],
        disabled=["Компонента"],
        hide_index=True,
        # height=423,
        width=300,
    )
    st.divider() 

    # Add threshold controls
    st.header("Настройки подбора")

    # Получаем дату по умолчанию из конфига
    default_date_str = config["model"]["stage_1"]["date_threshold"]
    try:
        # Пытаемся преобразовать строку из конфига в объект даты
        default_date = datetime.date.fromisoformat(default_date_str)
    except ValueError:
        # Если формат неверный, используем запасную дату
        logger.warning(f"Invalid date format in config for date_threshold: '{default_date_str}'. Using 2025-02-01.")
        default_date = datetime.date(2025, 2, 1)

    # Создаем виджет для выбора даты
    selected_date_threshold = st.date_input(
        "Рассматривать резюме не старше:",
        value=default_date,
        min_value=datetime.date(2015, 1, 1),
        max_value=datetime.date.today(),
        help="Резюме, опубликованные или обновленные до этой даты, будут отфильтрованы."
    )

    threshold = st.slider(
        "Минимальный процент соответствия",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Кандидаты с процентом соответствия ниже этого значения не будут показаны"
    )

    # st.radio(
    #     "Ограничение по расстоянию (км):",
    #     options=["Нет ограничений", "100", "500", "1000", "5000"], # Доступные опции
    #     key="distance_option", # Связываем с состоянием сессии
    #     horizontal=True, # Отображаем горизонтально
    #     help="Фильтрует кандидатов на 1-м этапе по расстоянию до вакансии."
    # )

    st.selectbox(
        label="Ограничение по расстоянию (км):",
        options=["Нет ограничений", "100", "500", "1000", "5000"], # Список доступных опций
        key="distance_option", # Связываем с состоянием сессии
        # index=distance_options.index(st.session_state.distance_option), # Можно задать индекс по умолчанию, если нужно
        help="Фильтрует кандидатов на 1-м этапе по расстоянию до вакансии."
    )

    # Add apply button for threshold
    # if st.button("Применить порог", key="apply_threshold"):
    #     st.session_state['current_threshold'] = threshold
    #     st.rerun()

vacancy = {}
placeholder_position = "Введите название должности, например, «Горнорабочий»."
vacancy["Должность"] = st.text_input(
    "Должность",
    placeholder=placeholder_position,
    value="",
    help=placeholder_position,
)
vacancy["Должность категория"] = vacancy["Должность"]
vacancy["Должность подкатегория"] = vacancy["Должность"]
placeholder_req = "Перечислите обязательные требования к кандидату, например ,уровень образования и предпочтительные специальности, объем опыта работы, физические способности для выполнения работы и тд."
vacancy["required"] = st.text_area(
    "Обязательные требования",
    placeholder=placeholder_req,
    value="",
    help=placeholder_req,
)
placeholder_loc = "Укажите адрес осуществления работы, тип занятости (вахтовый метод, полная занятость, частичная занятость и др.), а также необходимость переезда."
vacancy["location"] = st.text_area(
    "Локация и формат работы",
    placeholder=placeholder_loc,
    value="",
    help=placeholder_loc,
)
placeholder_add = "Перечислите необязательные требования к кандидату, например, специальные навыки, описание опыта работы, другие качества."
vacancy["additional"] = st.text_area(
    "Дополнительные требования",
    placeholder=placeholder_add,
    value="",
    help=placeholder_add,
)
# salary_feats = st.text_area("Уровень заработной платы", placeholder="Введите зарплатную вилку", value = None)

if st.button("Подобрать", type="primary"):
    if (
        vacancy["Должность"] != ""
        and vacancy["required"] != ""
        and vacancy["location"] != ""
    ):
        if mode == str(Mode.PROF):
            pass
        else:
            df_cv = load_data(f"./data_mass/candidates_new.csv")
            df_cv = df_cv.rename(columns={"address": "Адрес"})
        with st.status("Подготовка вакансии..."):
            vacancy = selector.preprocess_vacancy(vacancy)
        with st.status("Подбор кандидатов..."):
            if (
                not Path("./tmp_cvs.csv").exists()
                or config["general"]["mode"] == "prod"
            ):
                st.write(f"Первая фаза: анализ {df_cv.shape[0]} кандидатов..")
                
                ############### Ограничение по расстоянию ###############
                selected_option = st.session_state.get("distance_option", "Нет ограничений")
                max_distance_filter = None

                if selected_option != "Нет ограничений":
                    max_distance_filter = float(selected_option)

                ############### Обработка плашки ВАХТЫ ###############
                # Состояние галочки "Вахта"
                is_vahta = st.session_state.get("vahta_mode", False)
                
                save_first_stage_weights = config["model"]["stage_1"]["weights"]
                if is_vahta:
                    # Если галочка включена, устанавливаем вес "Адрес" в 0
                    save_first_stage_weights[1] = 0.0

                df_ranked_1st = selector.rank_first_stage(
                    vacancy=vacancy, df_relevant=df_cv.copy(),
                    date_threshold=selected_date_threshold, is_vahta=is_vahta, max_distance_filter=max_distance_filter,
                    first_stage_weights=np.array(save_first_stage_weights),
                )
                st.write(f"Вторая фаза: анализ {df_ranked_1st.shape[0]} кандидатов..")

                df_ranked_2nd, vacancy_prep, nan_mask = selector.rank_second_stage(
                    vacancy=vacancy,
                    df_relevant=df_ranked_1st.copy(),
                    df_weights=st.session_state["df_weights"],
                    score_threshold_stage_2=threshold
                )
                if config["general"]["mode"] != "prod":
                    df_ranked_2nd.to_csv("./tmp_cvs.csv", index=False)
                    with open("./tmp_vac.json", "w") as f:
                        json.dump(vacancy_prep, f, ensure_ascii=False)
                    np.save("./tmp_mask.npy", nan_mask)
            else:
                df_ranked_2nd = pd.read_csv("./tmp_cvs.csv")
                with open("./tmp_vac.json", "r") as f:
                    vacancy_prep = json.load(f)
                nan_mask = np.load("./tmp_mask.npy")

            data_cv = df2dict(df_ranked_2nd)
            st.session_state["computed"] = True
            st.write(f"Выбрано {df_ranked_2nd.shape[0]} лучших кандидатов.")
        if st.session_state["computed"]:
            if mode == str(Mode.PROF):
                nan_mask = np.delete(nan_mask, [1, 2, 5])
            st.subheader("Кандидаты", divider="blue")
            for key in data_cv:
                col1_results, col2_cv = st.columns(2)
                key_ = data_cv[key]["Должность"]
                if "(" in key and ")" not in key:
                    key_ += ")"
                key_ += f" ({round(data_cv[key]['sim_score_second'] * 100)}% match)"
                key_ = key + f" - {key_}"
                with st.expander(key_):
                    if mode == str(Mode.MASS):
                        url = f"https://www.avito.ru{data_cv[key]['link']}"
                        st.write(f"[Ссылка на Avito]({url})")
                    match_score_first = round(data_cv[key]["sim_score_first"] * 100)
                    accent_color = select_color(match_score_first)
                    st.markdown(
                        f"Первая фаза: :{accent_color}[{match_score_first}% match]"
                    )

                    match_score_second = round(data_cv[key]["sim_score_second"] * 100)
                    accent_color = select_color(match_score_second)
                    st.markdown(
                        f"Вторая фаза: :{accent_color}[{match_score_second}% match]"
                    )
                    if mode == str(Mode.PROF):
                        match_score_full_desc = round(
                            data_cv[key]["Full_description_sim"] * 100
                        )
                        accent_color = select_color(match_score_second)
                        st.markdown(
                            f"Похожесть по полному описанию: :{accent_color}[{match_score_full_desc}% match]"
                        )

                    ranking_features = deepcopy(
                        config["model"]["stage_2"]["ranking_features"]
                    )
                    if mode == str(Mode.PROF):
                        ranking_features.remove("Должность категория")
                        ranking_features.remove("Должность подкатегория")
                        ranking_features.remove("Full_description")
                        job_labels = [
                            "Должность_sim",
                            "Должность_cat_sim",
                            "Должность_subcat_sim",
                        ]
                        data_cv[key]["Должность_sim"] = (
                            sum([data_cv[key][job_label] for job_label in job_labels])
                            / 3
                        )
                    # else:
                    #     ranking_features.remove("date")
                    for i, feature in enumerate(ranking_features):
                        col_results_1, col_results_2, col_results_3 = st.columns(
                            [2, 1, 2], gap="small", vertical_alignment="center"
                        )
                        num_rows = (
                            max(len(data_cv[key][feature]), len(vacancy_prep[feature]))
                            / 20
                        )
                        container_height = round(num_rows * 30) + 60
                        with col_results_1:
                            if i == 0:
                                st.header("Кандидат")
                            container_cv = st.container(
                                border=True, height=container_height
                            )
                            feature_print = feature
                            if feature in map_names:
                                feature_print = map_names[feature]
                            container_cv.caption(feature_print)
                            if feature == "Адрес":
                                formated_text = format_intersection(
                                    vacancy_prep[feature],
                                    data_cv[key]["Адрес"],
                                )
                            else:
                                formated_text = format_intersection(
                                    vacancy_prep[feature],
                                    data_cv[key][feature],
                                )
                            container_cv.markdown(formated_text.capitalize())

                        with col_results_2:
                            if i == 0:
                                st.header(" ")
                            container_score = st.container(
                                border=True, height=container_height
                            )
                            match_score = round(data_cv[key][f"{feature}_sim"] * 100)
                            flag_vac = False
                            flag_cv = False
                            if nan_mask[i] == 0:
                                match_score = 0
                                flag_vac = True
                            if data_cv[key][feature].lower() in [
                                "нет данных",
                                "нет информации",
                                "",
                                "none",
                                "не указано",
                                "не указана",
                                "не указан",
                                "не задан",
                            ]:
                                match_score = 0
                                flag_cv = True
                            if flag_vac * flag_cv:
                                match_score = 0
                            accent_color = select_color(match_score)
                            container_score.markdown(
                                "<br>" * int((num_rows // 2)), unsafe_allow_html=True
                            )
                            if feature == "Адрес":
                                container_score.markdown(
                                    f":{accent_color}[{match_score}%\n близость]"
                                )
                            else:
                                if flag_vac:
                                    container_score.markdown(
                                        "Нет данных"
                                    )
                                else:
                                    container_score.markdown(
                                        f":{accent_color}[{match_score}%\nmatch]"
                                    )

                        with col_results_3:
                            if i == 0:
                                st.header("Вакансия")
                            container_vac = st.container(
                                border=True, height=container_height
                            )
                            feature_print = feature
                            if feature in map_names:
                                feature_print = map_names[feature]
                            container_vac.caption(feature_print)
                            if feature == "Адрес":
                                formated_text = format_intersection(
                                    data_cv[key]["Адрес"], vacancy_prep[feature]
                                )
                            else:
                                formated_text = format_intersection(
                                    data_cv[key][feature], vacancy_prep[feature]
                                )
                            container_vac.markdown(formated_text.capitalize())
                        if i < len(ranking_features) - 1:
                            st.divider()
    else:
        st.error(
            "Обязательные поля для заполнения: 'Должность', 'Обязательные требования', 'Локация и формат работы'",
            icon="🚨",
        )
