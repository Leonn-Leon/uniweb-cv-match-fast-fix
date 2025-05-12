import pandas as pd
import os
import pickle
from tqdm import tqdm
import time

from request_api import Embedder

# --- Конфигурация ---
INPUT_CSV_PATH = "data/processed/mass/candidates_hh.csv"
# OUTPUT_EMBEDDINGS_PATH = "data/processed/mass/candidate_embeddings.pkl" # Файл для сохранения структурированных эмбеддингов
CACHE_FILEPATH = "data/processed/mass/embeddings_cache.pkl" # Файл для кэша Embedder'а (текст -> эмбеддинг)

EMBEDDER_MODE = 'local'
MODEL_NAME = 'cointegrated/rubert-tiny2' # Ваша локальная модель
OPENAI_API_KEY = None # Не используется для локального режима

COLUMNS_TO_EMBED = [
    'Должность', 'address', 'Переезд', 'Образование',
    'Опыт работы', 'Компетенции', 'Тип занятости', 'График работы'
]

BATCH_SIZE = 64  # Уменьшите, если не хватает RAM/VRAM при параллельной обработке

# --- Основной скрипт ---
def main():
    print(f"Loading data from: {INPUT_CSV_PATH}")
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: File not found: {INPUT_CSV_PATH}"); return

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        total_rows = len(df)
        print(f"Loaded {total_rows} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}"); return

    actual_columns = [col for col in COLUMNS_TO_EMBED if col in df.columns]
    if len(actual_columns) != len(COLUMNS_TO_EMBED):
        missing = set(COLUMNS_TO_EMBED) - set(actual_columns)
        print(f"Warning: Columns not found and will be skipped: {missing}")
    if not actual_columns:
        print("Error: None of the specified columns found."); return

    print(f"Initializing Embedder (Mode: {EMBEDDER_MODE}, Model: {MODEL_NAME})")
    try:
        embedder = Embedder(  # No context manager here
            mode=EMBEDDER_MODE,
            model_name=MODEL_NAME,
            api_key=OPENAI_API_KEY, # Будет None для local
            cache_filepath=CACHE_FILEPATH
        )
        print(f"Initial cache size: {len(embedder.cache)}")
    except Exception as e:
        print(f"Error initializing Embedder: {e}"); return

    print("Collecting unique texts...")
    all_texts_to_embed = set()
    for col in actual_columns:
        unique_values = df[col].fillna('').astype(str).unique()
        all_texts_to_embed.update(val for val in unique_values if val)

    unique_texts_list = list(all_texts_to_embed)
    total_unique_texts = len(unique_texts_list)  # Сохраняем общее количество
    print(f"Found {total_unique_texts} unique non-empty texts to embed.")

    if not unique_texts_list:
        print("No texts to embed."); return

    print(f"Calculating embeddings (single-threaded, with batching)...")
    start_time = time.time()
    text_to_embedding_map = {}

    #  Обработка батчей и отображение прогресса
    for i in tqdm(range(0, len(unique_texts_list), BATCH_SIZE), desc="Embedding Progress"):
        batch = unique_texts_list[i:i + BATCH_SIZE]
        #  Вызываем embed_corpus для текущего батча (num_workers=1!)
        batch_embeddings = embedder.embed_corpus(
            batch,
            batch_size=BATCH_SIZE,
            num_workers=9 # Явное указание single-threaded режима
        )

        #  Обновляем отображение текст-эмбеддинг
        for text, emb in zip(batch, batch_embeddings):
            if emb is not None:
                text_to_embedding_map[text] = emb

    print(f"Embedding calculation finished in {time.time() - start_time:.2f} sec.")
    print(f"Successfully computed/cached {len(text_to_embedding_map)} embeddings. Final cache size: {len(embedder.cache)}")


if __name__ == "__main__":
    main()