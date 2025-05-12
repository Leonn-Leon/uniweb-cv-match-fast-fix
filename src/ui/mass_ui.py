# mass_ui.py
import streamlit as st
import numpy as np
from copy import deepcopy

from utils.utils import select_color, format_intersection
from utils.hardcode_data import map_names

def display_mass_input_form():
    """Отображает форму ввода для массового подбора и возвращает данные вакансии."""
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
    
    placeholder_req = "Перечислите обязательные требования к кандидату..."
    vacancy["required"] = st.text_area(
        "Обязательные требования",
        placeholder=placeholder_req,
        value="",
        help=placeholder_req,
    )
    
    placeholder_loc = "Укажите адрес осуществления работы..."
    vacancy["location"] = st.text_area(
        "Локация и формат работы",
        placeholder=placeholder_loc,
        value="",
        help=placeholder_loc,
    )
    
    placeholder_add = "Перечислите необязательные требования к кандидату..."
    vacancy["additional"] = st.text_area(
        "Дополнительные требования",
        placeholder=placeholder_add,
        value="",
        help=placeholder_add,
    )
    return vacancy

def display_mass_results(data_cv_dict, vacancy_prep, config, nan_mask):
    """Отображает результаты массового подбора."""
    st.subheader("Кандидаты", divider="blue")
    for key_original_id, cv_data in data_cv_dict.items():
        col1_results, col2_cv = st.columns(2)
        
        key_display_name = cv_data.get("Должность", "Нет должности")
        if "(" in key_original_id and ")" not in key_original_id:
            key_display_name += ")" 

        if "?" in key_original_id:
            key_original_id = key_original_id.split("?")[0][-10:]
        
        match_score_val = cv_data.get('sim_score_second', 0) * 100
        key_display = f"{key_original_id} - {key_display_name} ({round(match_score_val)}% match)"
        
        with st.expander(key_display):
            # url = f"https://www.avito.ru{cv_data.get('link', '')}"
            url = cv_data.get('link', '')
            st.write(f"[Ссылка]({url})")
            
            match_score_first = round(cv_data.get("sim_score_first", 0) * 100)
            accent_color = select_color(match_score_first)
            st.markdown(f"Первая фаза: :{accent_color}[{match_score_first}% match]")

            match_score_second = round(cv_data.get("sim_score_second", 0) * 100)
            accent_color = select_color(match_score_second)
            st.markdown(f"Вторая фаза: :{accent_color}[{match_score_second}% match]")

            ranking_features = deepcopy(config["model"]["stage_2"]["ranking_features"])

            for i, feature in enumerate(ranking_features):
                col_results_1, col_results_2, col_results_3 = st.columns(
                    [2, 1, 2], gap="small", vertical_alignment="center"
                )
                
                cv_feature_value = cv_data.get(feature, "Нет данных")
                vacancy_feature_value = vacancy_prep.get(feature, "Нет данных")

                # Проверяем, являются ли значения строками перед использованием len()
                cv_len = len(str(cv_feature_value)) if isinstance(cv_feature_value, (str, list, dict)) else 20 # default length
                vac_len = len(str(vacancy_feature_value)) if isinstance(vacancy_feature_value, (str, list, dict)) else 20 # default length

                num_rows = max(cv_len, vac_len) / 20.0
                container_height = round(num_rows * 30) + 60
                
                with col_results_1:
                    if i == 0:
                        st.header("Кандидат")
                    container_cv = st.container(border=True, height=container_height)
                    feature_print = feature_print = map_names.get(feature, feature)
                    container_cv.caption(feature_print)
                    
                    formated_text_cv = format_intersection(
                        str(vacancy_feature_value), str(cv_feature_value)
                    )
                    container_cv.markdown(formated_text_cv.capitalize())

                with col_results_2:
                    if i == 0:
                        st.header(" ")
                    container_score = st.container(border=True, height=container_height)
                    
                    match_score_feature = round(cv_data.get(f"{feature}_sim", 0) * 100)
                    flag_vac = False
                    flag_cv = False

                    if i < len(nan_mask) and nan_mask[i] == 0: # Проверка на выход за пределы nan_mask
                        match_score_feature = 0
                        flag_vac = True
                    
                    if str(cv_feature_value).lower() in [
                        "нет данных", "нет информации", "", "none", 
                        "не указано", "не указана", "не указан", "не задан"
                    ]:
                        match_score_feature = 0
                        flag_cv = True
                    
                    if flag_vac and flag_cv:
                        match_score_feature = 0
                    
                    accent_color_feature = select_color(match_score_feature)
                    container_score.markdown("<br>" * int((num_rows // 2)), unsafe_allow_html=True)
                    
                    display_feature_name = map_names.get(feature, feature)
                    if display_feature_name == "Адрес":
                        container_score.markdown(f":{accent_color_feature}[{match_score_feature}%\n близость]")
                    else:
                        if flag_vac:
                            container_score.markdown("Нет данных")
                        else:
                            container_score.markdown(f":{accent_color_feature}[{match_score_feature}%\nmatch]")
                
                with col_results_3:
                    if i == 0:
                        st.header("Вакансия")
                    container_vac = st.container(border=True, height=container_height)
                    feature_print_vac = map_names.get(feature, feature)
                    container_vac.caption(feature_print_vac)

                    formated_text_vac = format_intersection(
                        str(cv_feature_value), str(vacancy_feature_value)
                    )
                    container_vac.markdown(formated_text_vac.capitalize())
                
                if i < len(ranking_features) - 1:
                    st.divider()