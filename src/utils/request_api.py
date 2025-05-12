import concurrent
from typing import List, Dict, Optional, Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import os
import time
from loguru import logger

def process_corpus(
    corpus: List[str],
    func: callable,
    num_workers=8,
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, text_item) for text_item in corpus]

        with tqdm(total=len(corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

        results = []
        for future in futures:
            data = future.result()
            results.append(data)

        return results


class Embedder:
    """
    Gets text embeddings via API or local model with persistent caching.
    Uses ThreadPoolExecutor for parallel batch processing in both modes.
    """
    def __init__(self,
                 mode: str,
                 model_name: str,
                 api_key: Optional[str] = None,
                 cache_filepath: Optional[str] = "data/processed/mass/embeddings_cache.pkl"):

        if mode not in ['api', 'local']:
            raise ValueError("mode must be 'api' or 'local'")

        if mode == 'api':
            self.cache_filepath = ".".join(cache_filepath.split(".")[:-1]) + "_api.pkl"
            if not api_key: raise ValueError("api_key is required for 'api' mode")
            self.client = OpenAI(api_key=api_key)
            self._get_embeddings_for_batch = retry(
                wait=wait_random_exponential(min=1, max=40),
                stop=stop_after_attempt(10)
            )(self._get_api_embeddings)
            self._parallel_strategy = self._embed_corpus_threaded

        elif mode == 'local':
            
            self.cache_filepath = cache_filepath
            logger.info(f"Loading local model '{model_name}'...")
            start_load = time.time()
            self.client = SentenceTransformer(model_name)
            logger.info(f"Model loaded in {time.time() - start_load:.2f} seconds.")
            self._get_embeddings_for_batch = self._get_local_embeddings
            self._parallel_strategy = self._embed_corpus_threaded

        self.mode = mode
        self.model_name = model_name
        self.cache: Dict[str, List[float]] = self._load_cache()


    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_filepath):
            try:
                with open(self.cache_filepath, 'rb') as f:
                    loaded_cache = pickle.load(f)
                    logger.info(f"Loaded {len(loaded_cache)} items from cache: {self.cache_filepath}")
                    return loaded_cache
            except Exception as e:
                logger.error(f"Could not load cache from {self.cache_filepath}. Error: {e}")
        else:
            logger.warning(f"Cache file not found: {self.cache_filepath}. Starting empty.")
        return {}

    def _save_cache(self):
        if not self.cache_filepath:
            logger.warning("Cache filepath not set. Skipping save.")
            return
        logger.info(f"Saving {len(self.cache)} items to cache: {self.cache_filepath}...")
        try:
            cache_dir = os.path.dirname(self.cache_filepath)
            if cache_dir: os.makedirs(cache_dir, exist_ok=True)
            # Используем временный файл для большей надежности
            temp_filepath = self.cache_filepath + ".tmp"
            with open(temp_filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            os.replace(temp_filepath, self.cache_filepath) # Атомарная замена
            logger.info("Cache saved successfully.")
        except Exception as e:
            logger.error(f"Error saving cache to {self.cache_filepath}: {e}")

    def _get_api_embeddings(self, text_batch: List[str]) -> List[List[float]]:
        """Получает эмбеддинги через API OpenAI для одного батча."""
        if not text_batch: return []
        response = self.client.embeddings.create(input=text_batch, model=self.model_name).data
        return [data.embedding for data in response]

    def _get_local_embeddings(self, text_batch: List[str]) -> List[List[float]]:
        """Получает эмбеддинги локально для одного батча."""
        if not text_batch: return []
        # show_progress_bar=False, т.к. у нас будет общий прогресс-бар
        embeddings = self.client.encode(text_batch, show_progress_bar=False)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def _batchify(self, iterable: List[Any], n: int = 1):
        """Разбивает список на батчи."""
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    def _embed_corpus_threaded(self, texts_to_process: List[str], batch_size: int, num_workers: int) -> Dict[str, List[float]]:
        """Общая стратегия параллельной обработки с использованием ThreadPoolExecutor."""
        newly_computed_embeddings: Dict[str, List[float]] = {}
        if not texts_to_process:
            return newly_computed_embeddings

        batches = list(self._batchify(texts_to_process, batch_size))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Карта для сопоставления future с оригинальным батчем
            futures_map = {executor.submit(self._get_embeddings_for_batch, batch): batch for batch in batches}
            results_map: Dict[str, List[float]] = {}

            desc = f"Encoding Batches ({self.mode.upper()} Threads)"
            for future in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc=desc):
                original_batch = futures_map[future]
                try:
                    batch_embeddings = future.result()
                    # Сопоставляем результаты с исходными текстами батча
                    for text, embedding in zip(original_batch, batch_embeddings):
                         results_map[text] = embedding
                except Exception as e:
                    # Логируем ошибку для конкретного батча
                    logger.error(f"Batch failed ({self.mode}) for text starting with '{original_batch[0][:30]}...': {e}")
                    # Можно добавить логику для повторных попыток или присвоения None

            newly_computed_embeddings = results_map

        computed_count = len(newly_computed_embeddings)
        logger.info(f"{self.mode.capitalize()} encoding finished. Computed {computed_count} embeddings.")
        return newly_computed_embeddings

    def embed_corpus(
        self,
        corpus: List[str],
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> List[Optional[List[float]]]: # Возвращаем Optional для ясности
        """
        Рассчитывает эмбеддинги для корпуса текстов, используя кэш и ThreadPoolExecutor.
        """
        if not isinstance(corpus, list):
            raise TypeError("Corpus must be a list of strings.")

        final_embeddings: List[Optional[List[float]]] = [None] * len(corpus)
        texts_to_process_map: Dict[str, List[int]] = {}
        made_changes_to_cache = False

        # 1. Проверка кэша
        logger.info("Checking cache...")
        texts_in_cache_count = 0
        empty_texts_count = 0
        for i, text in enumerate(corpus):
            text_str = str(text) if text is not None else ""
            if not text_str:
                final_embeddings[i] = None # Обработка пустых/None текстов
                empty_texts_count += 1
                continue

            cached_embedding = self.cache.get(text_str)
            if cached_embedding is not None:
                final_embeddings[i] = cached_embedding
                texts_in_cache_count += 1
            else:
                if text_str not in texts_to_process_map:
                    texts_to_process_map[text_str] = []
                texts_to_process_map[text_str].append(i)

        unique_texts_to_process = list(texts_to_process_map.keys())
        logger.info(f"Cache hits: {texts_in_cache_count}. Empty/None texts: {empty_texts_count}. Texts to compute: {len(unique_texts_to_process)}")

        if not unique_texts_to_process:
            logger.info("No new texts to process.")
            return final_embeddings

        # 2. Вызов стратегии параллельной обработки
        start_compute_time = time.time()
        newly_computed = self._parallel_strategy(
            unique_texts_to_process, batch_size, num_workers
        )
        logger.info(f"Computation phase took {time.time() - start_compute_time:.2f} seconds.")

        # 3. Обновление кэша и результатов
        if newly_computed:
            logger.info(f"Updating cache with {len(newly_computed)} new embeddings...")
            self.cache.update(newly_computed)
            made_changes_to_cache = True

            for text, embedding in newly_computed.items():
                if text in texts_to_process_map:
                    for original_index in texts_to_process_map[text]:
                        final_embeddings[original_index] = embedding
        else:
            logger.warning("No new embeddings were computed.")

        # 4. Сохранение кэша
        if made_changes_to_cache:
            self._save_cache()

        # Проверка пропущенных (на случай ошибок в батчах)
        missing_count = sum(1 for i, emb in enumerate(final_embeddings) if emb is None and corpus[i] is not None and str(corpus[i]) != "")
        if missing_count > 0:
             logger.warning(f"Warning: {missing_count} non-empty texts still have no embeddings assigned (check for batch errors in logs).")

        return final_embeddings

    def generate_embeddings(self, df: pd.DataFrame, column_name: str, batch_size: int = 64, num_workers: int = 4) -> List[Optional[List[float]]]:
        """Обертка для получения эмбеддингов из колонки DataFrame."""
        texts = df[column_name].fillna('').astype(str).tolist()
        return self.embed_corpus(texts, batch_size=batch_size, num_workers=num_workers)