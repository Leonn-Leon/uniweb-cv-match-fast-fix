import requests
import streamlit as st
import time
import os
import toml

# --- Конфигурация ---
SECRETS_FILE_PATH = ".streamlit/secrets.toml" # Файл с секретами для этого теста

def load_test_secrets(filepath):
    """Загружает секреты из указанного TOML файла."""
    try:
        if os.path.exists(filepath):
            return toml.load(filepath)
        else:
            st.error(f"Файл секретов не найден: {filepath}")
            return {}
    except Exception as e:
        st.error(f"Ошибка загрузки файла секретов {filepath}: {e}")
        return {}

def get_test_proxies(secrets_data):
    """Формирует словарь прокси из данных секретов."""
    proxy_user = secrets_data.get("HUNTFLOW_PROXY_USER")
    proxy_pass = secrets_data.get("HUNTFLOW_PROXY_PASS")
    proxy_host = secrets_data.get("HUNTFLOW_PROXY_HOST")
    proxy_port = secrets_data.get("HUNTFLOW_PROXY_PORT")

    if all([proxy_user, proxy_pass, proxy_host, proxy_port]):
        proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        return {"http": proxy_url, "https": proxy_url}
    st.warning("Данные прокси не полностью определены в секретах.")
    return None

def attempt_token_refresh(secrets_data):
    """Пытается обновить токен Huntflow и выводит результат."""
    
    st.subheader("Попытка обновления токена Huntflow")

    refresh_token_value = secrets_data.get("HUNTFLOW_CURRENT_REFRESH_TOKEN")
    token_url = secrets_data.get("KEY_REFRESH_TOKEN_URL") 

    if not refresh_token_value:
        st.error("HUNTFLOW_CURRENT_REFRESH_TOKEN не найден в файле секретов.")
        return
    if not token_url:
        st.error("KEY_REFRESH_TOKEN_URL (URL для обновления токена) не найден в файле секретов.")
        st.info("Возможно, вам нужно явно указать его в переменной token_url в коде скрипта.")
        return

    st.write(f"URL для обновления токена: {token_url}")
    st.write(f"Используемый Refresh Token: ...{refresh_token_value[-10:]}") # Показываем только часть для безопасности

    payload = {
        'refresh_token': refresh_token_value,
    }
    
    st.write("Тело запроса (payload):")
    st.json(payload)

    # --- Формирование заголовков (headers) ---
    headers = {
        'Content-Type': 'application/json' 
    }

    st.write("Заголовки запроса (headers):")
    st.json(headers)

    proxies = get_test_proxies(secrets_data)
    if proxies:
        st.write("Используемые прокси:")
        st.json(proxies)
    else:
        st.warning("Запрос будет выполнен без прокси (если они не настроены).")

    if st.button("Обновить токен"):
        with st.spinner("Отправка запроса на обновление токена..."):
            try:
                response = requests.post(token_url, json=payload, headers=headers, proxies=proxies)
                
                st.write(f"Статус ответа: {response.status_code}")
                
                try:
                    response_data = response.json()
                    st.write("Ответ от сервера (JSON):")
                    st.json(response_data)

                    if response.ok: # Статус 2xx
                        new_access_token = response_data.get('access_token')
                        new_refresh_token = response_data.get('refresh_token')
                        expires_in = response_data.get('expires_in')

                        if new_access_token:
                            st.success(f"Новый Access Token получен: ...{new_access_token[-10:]}")
                            if expires_in:
                                st.info(f"Срок действия: {expires_in} секунд (примерно {expires_in/3600:.2f} часов)")
                            if new_refresh_token:
                                st.info(f"Новый Refresh Token получен: ...{new_refresh_token[-10:]} (сохраните его, если он отличается!)")
                            else:
                                st.warning("Новый Refresh Token не был возвращен (возможно, старый все еще действителен).")
                        else:
                            st.error("Access token не найден в успешном ответе.")
                    else:
                        st.error("Запрос на обновление токена не был успешным (см. детали выше).")

                except ValueError: # Если ответ не JSON
                    st.error("Ответ от сервера не в формате JSON.")
                    st.text(response.text)
                
            except requests.exceptions.ProxyError as e:
                 st.error(f"Ошибка прокси при обновлении токена: {e}")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка сети при обновлении токена: {e}")
            except Exception as e:
                st.error(f"Неожиданная ошибка при обновлении токена: {e}")

if __name__ == "__main__":
    st.set_page_config(page_title="Тест обновления токена Huntflow")
    st.title("Тестирование обновления токена Huntflow")

    st.info(f"Этот скрипт попытается обновить токен Huntflow, используя данные из файла: '{SECRETS_FILE_PATH}'. "
            "Убедитесь, что файл существует и содержит правильные значения, особенно HUNTFLOW_CURRENT_REFRESH_TOKEN' "
            "и KEY_REFRESH_TOKEN_URL (или URL указан в коде).")
    
    st.warning("ВАЖНО: Адаптируйте формирование `payload` и `headers` в функции `attempt_token_refresh` "
               "в соответствии с документацией Huntflow API по OAuth 2.0 / обновлению токенов!")

    secrets_data = load_test_secrets(SECRETS_FILE_PATH)

    if secrets_data:
        attempt_token_refresh(secrets_data)
    else:
        st.error("Не удалось загрузить данные из файла секретов. Проверьте путь и содержимое файла.")