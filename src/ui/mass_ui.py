# mass_ui.py
import streamlit as st
import numpy as np # Не используется в display_mass_input_form, но оставлю, если нужен для display_mass_results
from copy import deepcopy # Не используется в display_mass_input_form, но оставлю

# Предполагается, что эти утилиты используются в display_mass_results
from src.utils.utils import select_color, format_intersection 
from src.utils.hardcode_data import map_names

def display_mass_input_form():
    """Отображает форму ввода для массового подбора и возвращает данные вакансии,
    используя st.session_state для предзаполнения."""
    
    vacancy = {}
    
    # Ключи должны совпадать с теми, что используются в app.py для st.session_state
    key_title = "vacancy_form_title"
    key_required = "vacancy_form_required"
    key_location = "vacancy_form_location"
    key_additional = "vacancy_form_optional"

    placeholder_position = "Введите название должности, например, «Горнорабочий»."
    vacancy["Должность"] = st.text_input(
        "Должность",
        placeholder=placeholder_position,
        value=st.session_state.get(key_title, ""), # Берем значение из session_state
        key=key_title, # Присваиваем ключ
        help=placeholder_position,
    )
    vacancy["Должность категория"] = vacancy["Должность"] 
    vacancy["Должность подкатегория"] = vacancy["Должность"]
    
    placeholder_req = "Перечислите обязательные требования к кандидату..."
    vacancy["required"] = st.text_area(
        "Обязательные требования",
        placeholder=placeholder_req,
        value=st.session_state.get(key_required, ""), # Берем значение из session_state
        key=key_required, # Присваиваем ключ
        help=placeholder_req,
    )
    
    placeholder_loc = "Укажите адрес осуществления работы..."
    vacancy["location"] = st.text_area(
        "Локация и формат работы",
        placeholder=placeholder_loc,
        value=st.session_state.get(key_location, ""), # Берем значение из session_state
        key=key_location, # Присваиваем ключ
        help=placeholder_loc,
    )
    
    placeholder_add = "Перечислите необязательные требования к кандидату..."
    vacancy["additional"] = st.text_area(
        "Дополнительные требования",
        placeholder=placeholder_add,
        value=st.session_state.get(key_additional, ""), # Берем значение из session_state
        key=key_additional, # Присваиваем ключ
        help=placeholder_add,
    )
    
    return {
        "Должность": st.session_state.get(key_title, ""),
        "Должность категория": st.session_state.get(key_title, ""), # По-прежнему зависит от основного поля "Должность"
        "Должность подкатегория": st.session_state.get(key_title, ""), # По-прежнему зависит от основного поля "Должность"
        "required": st.session_state.get(key_required, ""),
        "location": st.session_state.get(key_location, ""),
        "additional": st.session_state.get(key_additional, "")
    }

def display_mass_results(data_cv_dict, vacancy_prep, config, nan_mask):
    """Отображает результаты массового подбора. (Эта функция остается без изменений)"""
    st.subheader("Кандидаты", divider="blue")
    for key_original_id, cv_data in data_cv_dict.items():
        # col1_results, col2_cv = st.columns(2) # Эта строка была закомментирована или удалена, оставим так
        
        key_display_name = cv_data.get("Должность", "Нет должности")
        
        match_score_val = cv_data.get('sim_score_second', cv_data.get('Итоговый балл', 0))
        if isinstance(match_score_val, (float, int)) and not isinstance(match_score_val, bool):
             match_score_val_perc = round(match_score_val * 100 if match_score_val <= 1.0 else match_score_val)
        else:
            match_score_val_perc = 0


        key_display = f"{key_original_id} - {key_display_name} ({match_score_val_perc}% match)"
        
        with st.expander(key_display):
            url = cv_data.get('link', '')
            if url: # Проверяем, что ссылка есть
                st.write(f"[Ссылка на резюме]({url})") # Изменил текст ссылки для ясности
            else:
                st.write("Ссылка на резюме отсутствует.")
            
            match_score_first_raw = cv_data.get("sim_score_first", 0)
            match_score_first = round(match_score_first_raw * 100 if isinstance(match_score_first_raw, float) and match_score_first_raw <=1 else match_score_first_raw)
            accent_color = select_color(match_score_first)
            st.markdown(f"Первая фаза: :{accent_color}[{match_score_first}% match]")

            match_score_second_raw = cv_data.get("sim_score_second", 0)
            match_score_second = round(match_score_second_raw * 100 if isinstance(match_score_second_raw, float) and match_score_second_raw <=1 else match_score_second_raw)
            accent_color_second = select_color(match_score_second) # Использовал новую переменную для цвета второго этапа
            st.markdown(f"Вторая фаза: :{accent_color_second}[{match_score_second}% match]")

            # Предполагаем, что config и ranking_features корректно передаются
            ranking_features = deepcopy(config.get("model", {}).get("stage_2", {}).get("ranking_features", []))

            for i, feature in enumerate(ranking_features):
                col_results_1, col_results_2, col_results_3 = st.columns(
                    [2, 1, 2], gap="small", vertical_alignment="center"
                )
                
                cv_feature_value = cv_data.get(feature, "Нет данных")
                vacancy_feature_value = vacancy_prep.get(feature, "Нет данных")

                # Проверяем, являются ли значения строками перед использованием len()
                cv_len = len(str(cv_feature_value)) if isinstance(cv_feature_value, (str, list, dict)) else 20 
                vac_len = len(str(vacancy_feature_value)) if isinstance(vacancy_feature_value, (str, list, dict)) else 20 

                num_rows = max(cv_len, vac_len) / 20.0 
                container_height = round(num_rows * 30) + 60
                
                with col_results_1:
                    if i == 0:
                        st.markdown("##### Кандидат") # Изменил на markdown для единообразия
                    container_cv = st.container(border=True, height=container_height)
                    feature_print_cv = map_names.get(feature, feature) # Исправил опечатку feature_print = feature_print
                    container_cv.caption(feature_print_cv)
                    
                    formated_text_cv = format_intersection(
                        str(vacancy_feature_value), str(cv_feature_value)
                    )
                    container_cv.markdown(formated_text_cv.capitalize())

                with col_results_2:
                    if i == 0:
                        st.markdown("#####  ") # Пустой заголовок для выравнивания
                    container_score = st.container(border=True, height=container_height)
                    
                    match_score_feature_raw = cv_data.get(f"{feature}_sim", 0)
                    match_score_feature = round(match_score_feature_raw * 100 if isinstance(match_score_feature_raw, float) and match_score_feature_raw <=1 else match_score_feature_raw)

                    flag_vac = False
                    flag_cv = False

                    if i < len(nan_mask) and nan_mask[i] == 0: 
                        match_score_feature = 0
                        flag_vac = True
                    
                    if str(cv_feature_value).lower().strip() in [
                        "нет данных", "нет информации", "", "none", 
                        "не указано", "не указана", "не указан", "не задан"
                    ]:
                        match_score_feature = 0
                        flag_cv = True
                    
                    if flag_vac and flag_cv: # Если нет данных ни у вакансии, ни у кандидата
                        match_score_feature = 0 # или можно отобразить "N/A"
                    
                    accent_color_feature = select_color(match_score_feature)
                    
                    display_feature_name_map = map_names.get(feature, feature) # Исправил имя переменной
                    if display_feature_name_map == "Адрес":
                        container_score.markdown(f":{accent_color_feature}[{match_score_feature}%\n близость]")
                    else:
                        if flag_vac and not flag_cv: # Нет данных в вакансии
                             container_score.markdown("Нет данных в вакансии")
                        elif flag_cv and not flag_vac: # Нет данных у кандидата
                             container_score.markdown("Нет данных у кандидата")
                        elif flag_cv and flag_vac: # Нет данных нигде
                             container_score.markdown("Нет данных")
                        else:
                            container_score.markdown(f":{accent_color_feature}[{match_score_feature}%\nmatch]")
                
                with col_results_3:
                    if i == 0:
                        st.markdown("##### Вакансия")
                    container_vac = st.container(border=True, height=container_height)
                    feature_print_vac = map_names.get(feature, feature)
                    container_vac.caption(feature_print_vac)

                    formated_text_vac = format_intersection(
                        str(cv_feature_value), str(vacancy_feature_value)
                    )
                    container_vac.markdown(formated_text_vac.capitalize())
                
                if i < len(ranking_features) - 1:
                    st.divider()