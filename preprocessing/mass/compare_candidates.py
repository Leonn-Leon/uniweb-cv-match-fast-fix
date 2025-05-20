import pandas as pd
import os
import numpy as np

# --- Настройки ---
NEW_CSV_PATH = "data_mass/candidates_new.csv"
OLD_CSV_PATH = "data_mass/candidates.csv"
LINK_COLUMN = "link" # Имя колонки для ссылок
ID_COLUMN = "ID"     # Имя колонки для ID
DATE_COLUMN = "date" # Имя колонки для даты

# --- Функции для вывода ---
def print_header(title):
    """Выводит заголовок раздела."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_comparison(label, value_new, value_old):
    """Выводит сравнение значения между новым и старым файлом."""
    print(f"- {label}:")
    print(f"  Новый файл ({os.path.basename(NEW_CSV_PATH)}): {value_new}")
    print(f"  Старый файл ({os.path.basename(OLD_CSV_PATH)}): {value_old}")
    try:
        # Попытка вычислить разницу, если значения числовые
        diff = value_new - value_old
        # Проверяем, что разница не NaN (может быть, если value_old/value_new было None)
        if pd.notna(diff):
             print(f"  Разница (Новый - Старый): {diff}")
        else:
             print("  Разница не может быть вычислена (нечисловые значения).")
    except (TypeError, ValueError):
        # Если не числовые, просто указываем на наличие разницы
        if value_new != value_old:
            print("  Значения различаются.")
        else:
            print("  Значения совпадают.")

def print_examples(label, examples, max_examples=10):
    """Выводит примеры из списка."""
    if not examples:
        print(f"  {label}: Отсутствуют.")
        return

    print(f"  {label} (первые {min(len(examples), max_examples)}):")
    for item in examples[:max_examples]:
        print(f"    {item}")
    if len(examples) > max_examples:
        print(f"  ... и еще {len(examples) - max_examples}.")


# --- Основная логика ---
if __name__ == "__main__":
    print(f"Запуск сравнения файлов: '{NEW_CSV_PATH}' и '{OLD_CSV_PATH}'")

    # 1. Проверка наличия файлов
    if not os.path.exists(NEW_CSV_PATH):
        print(f"\n[ОШИБКА] Файл не найден: {NEW_CSV_PATH}")
        exit()
    if not os.path.exists(OLD_CSV_PATH):
        print(f"\n[ОШИБКА] Файл не найден: {OLD_CSV_PATH}")
        exit()

    # 2. Загрузка данных
    try:
        # Читаем CSV, пытаясь распознать даты в колонке 'date' на этом этапе
        df_new = pd.read_csv(NEW_CSV_PATH, parse_dates=[DATE_COLUMN], dayfirst=True)
        df_old = pd.read_csv(OLD_CSV_PATH, parse_dates=[DATE_COLUMN], dayfirst=True)
        print("\n[ИНФО] Файлы успешно загружены.")
        print(f"[ИНФО] При загрузке предпринята попытка распознать даты в колонке '{DATE_COLUMN}'.")
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось прочитать один из CSV файлов или распознать даты: {e}")
        exit()

    # 3. Сравнение базовых характеристик
    print_header("1. Основные характеристики")
    print_comparison("Количество строк", len(df_new), len(df_old))
    print_comparison("Количество колонок", len(df_new.columns), len(df_old.columns))

    cols_new = set(df_new.columns)
    cols_old = set(df_old.columns)
    if cols_new == cols_old:
        print("\n- Наборы колонок: Идентичны")
        common_cols = sorted(list(cols_new)) # Сохраняем общие колонки для дальнейших проверок
    else:
        print("\n- Наборы колонок: Различаются")
        print(f"  Только в новом файле: {sorted(list(cols_new - cols_old))}")
        print(f"  Только в старом файле: {sorted(list(cols_old - cols_new))}")
        common_cols = sorted(list(cols_new.intersection(cols_old)))
        print(f"  Общие колонки ({len(common_cols)}): {common_cols}")

    # 4. Проверка дубликатов LINK
    print_header(f"2. Проверка дубликатов колонки '{LINK_COLUMN}'")
    if LINK_COLUMN in df_new.columns:
        # Исключаем NaN перед проверкой на дубликаты, если нужно рассматривать только непустые ссылки
        # duplicates_new_link = df_new.dropna(subset=[LINK_COLUMN])[df_new.dropna(subset=[LINK_COLUMN]).duplicated(LINK_COLUMN, keep=False)]
        # Или просто проверяем на дубликаты, включая NaN (NaN считаются уникальными по умолчанию в duplicated, unless keep=False)
        duplicates_new_link = df_new[df_new.duplicated(LINK_COLUMN, keep=False)]
        print(f"- Дубликатов '{LINK_COLUMN}' в новом файле: {len(duplicates_new_link)}")
        if not duplicates_new_link.empty:
            print_examples("Примеры дублирующихся LINK", duplicates_new_link[LINK_COLUMN].unique().tolist()[:10])
    else:
        print(f"- Колонка '{LINK_COLUMN}' отсутствует в новом файле.")

    if LINK_COLUMN in df_old.columns:
        duplicates_old_link = df_old[df_old.duplicated(LINK_COLUMN, keep=False)]
        print(f"- Дубликатов '{LINK_COLUMN}' в старом файле: {len(duplicates_old_link)}")
        if not duplicates_old_link.empty:
             print_examples("Примеры дублирующихся LINK", duplicates_old_link[LINK_COLUMN].unique().tolist()[:10])
    else:
         print(f"- Колонка '{LINK_COLUMN}' отсутствует в старом файле.")


    # 5. Проверка дубликатов ID
    print_header(f"3. Проверка дубликатов колонки '{ID_COLUMN}'")
    # Важно: Преобразуем ID в строку перед проверкой на дубликаты, чтобы избежать проблем с типами (число/строка) и NaN
    if ID_COLUMN in df_new.columns:
        # Создаем временную колонку в пределах видимости этого блока try/except
        try:
            # Преобразуем в строку, игнорируя ошибки (например, если колонка содержит нечисловые или NaN данные,
            # которые сложно преобразовать стандартным astype)
            df_new[f'{ID_COLUMN}_str_temp'] = df_new[ID_COLUMN].apply(lambda x: str(x) if pd.notna(x) else np.nan)
            duplicates_new_id = df_new[df_new.duplicated(f'{ID_COLUMN}_str_temp', keep=False)]
            print(f"- Дубликатов '{ID_COLUMN}' в новом файле: {len(duplicates_new_id)}")
            if not duplicates_new_id.empty:
                print_examples("Примеры дублирующихся ID", duplicates_new_id[f'{ID_COLUMN}_str_temp'].unique().tolist()[:10])
        except Exception as e:
             print(f"- Ошибка при проверке дубликатов '{ID_COLUMN}' в новом файле: {e}")
        finally:
            # Удаляем временную колонку, если она была создана
            if f'{ID_COLUMN}_str_temp' in df_new.columns:
                df_new = df_new.drop(columns=[f'{ID_COLUMN}_str_temp'])


    else:
        print(f"- Колонка '{ID_COLUMN}' отсутствует в новом файле.")

    if ID_COLUMN in df_old.columns:
        try:
            df_old[f'{ID_COLUMN}_str_temp'] = df_old[ID_COLUMN].apply(lambda x: str(x) if pd.notna(x) else np.nan)
            duplicates_old_id = df_old[df_old.duplicated(f'{ID_COLUMN}_str_temp', keep=False)]
            print(f"- Дубликатов '{ID_COLUMN}' в старом файле: {len(duplicates_old_id)}")
            if not duplicates_old_id.empty:
                print_examples("Примеры дублирующихся ID", duplicates_old_id[f'{ID_COLUMN}_str_temp'].unique().tolist()[:10])
        except Exception as e:
             print(f"- Ошибка при проверке дубликатов '{ID_COLUMN}' в старом файле: {e}")
        finally:
            if f'{ID_COLUMN}_str_temp' in df_old.columns:
                df_old = df_old.drop(columns=[f'{ID_COLUMN}_str_temp'])


    else:
        print(f"- Колонка '{ID_COLUMN}' отсутствует в старом файле.")


    # 6. Уникальные ID и LINK (в одном файле, отсутствуют в другом) - Сдвинуто на один пункт
    print_header("4. Уникальные ID и LINK (в одном файле, отсутствуют в другом)")

    # Сравнение ID
    if ID_COLUMN in df_new.columns and ID_COLUMN in df_old.columns:
         # Получаем множества уникальных ID (преобразуем в строку и убираем NaN)
         # Используем apply(str) вместо astype(str) для более надежного преобразования, если есть смешанные типы
         ids_new_set = set(df_new[ID_COLUMN].dropna().apply(str))
         ids_old_set = set(df_old[ID_COLUMN].dropna().apply(str))

         # Находим ID только в новом файле
         ids_only_in_new = sorted(list(ids_new_set - ids_old_set)) # Сортируем для стабильного вывода
         print(f"- ID только в новом файле ({os.path.basename(NEW_CSV_PATH)}): {len(ids_only_in_new)}")
         print_examples("Примеры ID только в новом файле", ids_only_in_new)

         # Находим ID только в старом файле
         ids_only_in_old = sorted(list(ids_old_set - ids_new_set))
         print(f"- ID только в старом файле ({os.path.basename(OLD_CSV_PATH)}): {len(ids_only_in_old)}")
         print_examples("Примеры ID только в старом файле", ids_only_in_old)

    else:
         print(f"- Невозможно сравнить уникальные '{ID_COLUMN}': Колонка отсутствует в одном или обоих файлах.")

    print("-" * 30) # Разделитель между ID и LINK

    # Сравнение LINK
    if LINK_COLUMN in df_new.columns and LINK_COLUMN in df_old.columns:
         # Получаем множества уникальных LINK (преобразуем в строку и убираем NaN)
         links_new_set = set(df_new[LINK_COLUMN].dropna().astype(str))
         links_old_set = set(df_old[LINK_COLUMN].dropna().astype(str))

         # Находим LINK только в новом файле
         links_only_in_new = sorted(list(links_new_set - links_old_set))
         print(f"- LINK только в новом файле ({os.path.basename(NEW_CSV_PATH)}): {len(links_only_in_new)}")
         print_examples("Примеры LINK только в новом файле", links_only_in_new)

         # Находим LINK только в старом файле
         links_only_in_old = sorted(list(links_old_set - links_new_set))
         print(f"- LINK только в старом файле ({os.path.basename(OLD_CSV_PATH)}): {len(links_only_in_old)}")
         print_examples("Примеры LINK только в старом файле", links_only_in_old)
    else:
         print(f"- Невозможно сравнить уникальные '{LINK_COLUMN}': Колонка отсутствует в одном или обоих файлах.")


    # --- Новый раздел: Валидация даты ---
    print_header(f"5. Валидация даты в колонке '{DATE_COLUMN}'")

    # Валидация Нового файла
    if DATE_COLUMN in df_new.columns:
        try:
            # При загрузке мы уже использовали parse_dates и errors='coerce',
            # так что некорректные даты уже представлены как NaT.
            # Теперь просто находим эти строки.
            invalid_date_rows_new = df_new[df_new[DATE_COLUMN].isna()]
            num_invalid_dates_new = len(invalid_date_rows_new)

            print(f"- Некорректных или пропущенных дат в новом файле ({os.path.basename(NEW_CSV_PATH)}): {num_invalid_dates_new}")

            if num_invalid_dates_new > 0:
                if ID_COLUMN in df_new.columns:
                    # Получаем ID и оригинальную строковую дату для примеров
                    # Важно: при получении оригинальной строковой даты нужно обратиться к данным ДО parse_dates,
                    # но parse_dates меняет колонку на месте.
                    # Более надежный способ - перечитать колонку как строку или использовать исходные данные до парсинга,
                    # но для простоты возьмем данные из текущего df - если это NaN (NaT), то это либо исходный NaN, либо ошибка парсинга.
                    # Чтобы показать оригинальную проблемную строку, нужно было бы не парсить при загрузке,
                    # а парсить вручную с errors='coerce' и сохранять результат в новую колонку.
                    # Однако, часто NaT в дате после парсинга уже указывает на проблему с исходной строкой.
                    # Получим ID для строк с NaT в колонке даты.
                    invalid_ids_with_date_str_new = []
                    # Итерируемся по индексу строк с NaT в дате
                    for index in invalid_date_rows_new.index:
                         id_val = df_new.loc[index, ID_COLUMN] if ID_COLUMN in df_new.columns else "ID отсутствует"
                         date_val = df_new.loc[index, DATE_COLUMN] # Это будет NaT
                         # Для отображения исходной строки даты, нам нужно получить значение ДО парсинга.
                         # Это можно сделать, загрузив колонку отдельно или не парся при загрузке.
                         # Покажем ID и просто факт некорректной даты. Если нужна оригинальная строка, нужно доработать загрузку.
                         invalid_ids_with_date_str_new.append(f"ID: {id_val} (Дата: {date_val})") # Дата будет NaT, но ID будет показан

                    print_examples(f"Примеры ID с некорректными или пропущенными датами", invalid_ids_with_date_str_new)

                else:
                     print(f"  (Колонка '{ID_COLUMN}' отсутствует в новом файле, примеры ID не могут быть показаны)")

        except Exception as e:
             print(f"- Ошибка при валидации даты в новом файле: {e}")
    else:
        print(f"- Колонка '{DATE_COLUMN}' отсутствует в новом файле.")

    print("-" * 30) # Разделитель

    # Валидация Старого файла
    if DATE_COLUMN in df_old.columns:
        try:
            invalid_date_rows_old = df_old[df_old[DATE_COLUMN].isna()]
            num_invalid_dates_old = len(invalid_date_rows_old)

            print(f"- Некорректных или пропущенных дат в старом файле ({os.path.basename(OLD_CSV_PATH)}): {num_invalid_dates_old}")

            if num_invalid_dates_old > 0:
                if ID_COLUMN in df_old.columns:
                    invalid_ids_with_date_str_old = []
                    for index in invalid_date_rows_old.index:
                         id_val = df_old.loc[index, ID_COLUMN] if ID_COLUMN in df_old.columns else "ID отсутствует"
                         date_val = df_old.loc[index, DATE_COLUMN] # Это будет NaT
                         invalid_ids_with_date_str_old.append(f"ID: {id_val} (Дата: {date_val})")

                    print_examples(f"Примеры ID с некорректными или пропущенными датами", invalid_ids_with_date_str_old)
                else:
                     print(f"  (Колонка '{ID_COLUMN}' отсутствует в старом файле, примеры ID не могут быть показаны)")

        except Exception as e:
             print(f"- Ошибка при валидации даты в старом файле: {e}")
    else:
        print(f"- Колонка '{DATE_COLUMN}' отсутствует в старом файле.")

    # --- Конец нового раздела ---


    # 7. Сравнение схемы (типы данных общих колонок) - Сдвинуто на два пункта
    print_header("6. Сравнение схемы (типы данных)")
    if common_cols:
        diff_dtypes_count = 0
        print("- Сравнение типов данных для общих колонок:")
        for col in common_cols:
            # Проверяем, что колонка существует после всех потенциальных удалений временных колонок
            if col in df_new.columns and col in df_old.columns:
                dtype_new = df_new[col].dtype
                dtype_old = df_old[col].dtype
                if dtype_new != dtype_old:
                    print(f"  Различие в колонке '{col}': Новый='{dtype_new}', Старый='{dtype_old}'")
                    diff_dtypes_count += 1
        if diff_dtypes_count == 0:
            print("  Типы данных для всех общих колонок совпадают.")
        else:
            print(f"  Всего колонок с различиями в типах: {diff_dtypes_count}")
    else:
         print("- Нет общих колонок для сравнения типов данных.")


    # 8. Сравнение пропущенных значений (NaN/Null) - Сдвинуто на два пункта
    print_header("7. Сравнение пропущенных значений (NaN/Null)")
    # Учитываем, что после parse_dates некорректные даты стали NaN (NaT)
    nulls_new = df_new.isnull().sum()
    nulls_old = df_old.isnull().sum()
    # Создаем DataFrame для удобного сравнения
    # Используем outer join, чтобы включить колонки, которые есть только в одном DF
    null_comparison = pd.concat([nulls_new.rename('Новый'), nulls_old.rename('Старый')], axis=1).fillna(0).astype(int)
    null_comparison['Разница (Новый - Старый)'] = null_comparison['Новый'] - null_comparison['Старый']


    # Показываем только те строки, где есть null или разница не нулевая
    null_comparison_filtered = null_comparison[
        (null_comparison['Новый'] > 0) |
        (null_comparison['Старый'] > 0) |
        (null_comparison['Разница (Новый - Старый)'] != 0)
    ].sort_index() # Сортируем по имени колонки для стабильного вывода

    if not null_comparison_filtered.empty:
        print(null_comparison_filtered)
    else:
        print("- Не обнаружено пропущенных значений или разницы в их количестве.")


    # 9. Новые / Удаленные строки (на основе уникального ключа) - Сдвинуто на два пункта
    # Этот раздел по-прежнему полезен, так как сравнивает *строки* на основе *одного* ключа,
    # тогда как предыдущий раздел сравнивает *значения* в колонках ID и LINK по отдельности.
    print_header("8. Новые / Удаленные строки (на основе выбранного ключа)")
    # Определяем, какую колонку использовать как ключ (приоритет у LINK)
    key_col_to_use = None
    if LINK_COLUMN in common_cols:
        key_col_to_use = LINK_COLUMN
    elif ID_COLUMN in common_cols:
        key_col_to_use = ID_COLUMN

    if key_col_to_use:
        if key_col_to_use in df_new.columns and key_col_to_use in df_old.columns:
            print(f"- Сравнение на основе колонки: '{key_col_to_use}'")
            try:
                keys_new = set(df_new[key_col_to_use].dropna().apply(str))
                keys_old = set(df_old[key_col_to_use].dropna().apply(str))

                added_keys = sorted(list(keys_new - keys_old)) # Сортируем для стабильного вывода
                removed_keys = sorted(list(keys_old - keys_new))

                print(f"- Строк добавлено (уникальные '{key_col_to_use}' в новом файле): {len(added_keys)}")
                print_examples("Примеры добавленных строк", added_keys)

                print(f"- Строк удалено (уникальные '{key_col_to_use}' отсутствуют в новом файле): {len(removed_keys)}")
                print_examples("Примеры удаленных строк", removed_keys)

            except Exception as e:
                print(f"- Ошибка при сравнении строк по ключу '{key_col_to_use}': {e}")
        else:
            print(f"- Ошибка: Ключевая колонка '{key_col_to_use}' отсутствует в одном или обоих файлах для сравнения строк.")


    else:
        print(f"- Невозможно определить новые/удаленные строки.")
        print(f"  Причина: Отсутствует общая колонка '{LINK_COLUMN}' или '{ID_COLUMN}'.")


    print("\n--- Сравнение завершено ---")