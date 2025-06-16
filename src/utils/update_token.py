import requests
from loguru import logger

def _prepare_proxies(proxy_details: dict):
    proxy_user = proxy_details.get("PROXY_USER")
    proxy_pass = proxy_details.get("PROXY_PASS")
    proxy_host = proxy_details.get("PROXY_HOST")
    proxy_port = proxy_details.get("PROXY_PORT")

    if all([proxy_user, proxy_pass, proxy_host, proxy_port]):
        proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        return {"http": proxy_url, "https": proxy_url}
    return None

def refresh_api_token(token_url: str, refresh_token_value: str, proxy_details: dict = None):
    if not token_url or not refresh_token_value:
        return None

    payload = {
        'refresh_token': refresh_token_value,
    }
    headers = {
        'Content-Type': 'application/json'
    }

    proxies = _prepare_proxies(proxy_details) if proxy_details else None

    try:
        response = requests.post(token_url, json=payload, headers=headers, proxies=proxies, timeout=5)
        response.raise_for_status()

        response_data = response.json()
        # new_access_token = response_data.get('access_token')

        # if new_access_token:
        #     return new_access_token
        logger.info("Вот что получили при обновлении токена: "+str(response_data))
        return response_data

    except Exception as e:
        logger.info(f"Failed to refresh API token:{e}, response: {response.json()}")
        return None
    
if __name__ == "__main__":
    import streamlit as st
    token_url = st.secrets["HUNTFLOW_REFRESH_TOKEN_URL"]
    refresh_token_value = st.secrets["HUNTFLOW_CURRENT_REFRESH_TOKEN"]
    proxy_details = {
        "PROXY_USER": st.secrets.get("HUNTFLOW_PROXY_USER"),
        "PROXY_PASS": st.secrets.get("HUNTFLOW_PROXY_PASS"),
        "PROXY_HOST": st.secrets.get("HUNTFLOW_PROXY_HOST"),
        "PROXY_PORT": st.secrets.get("HUNTFLOW_PROXY_PORT")
    }
    logger.info(refresh_api_token(token_url, refresh_token_value, proxy_details))