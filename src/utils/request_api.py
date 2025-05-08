import concurrent
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


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


class OpenAIEmbedder:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
    def get_embeddings(self, input: List):
        response = self.client.embeddings.create(
            input=input, model=self.model_name
        ).data
        return [data.embedding for data in response]

    def batchify(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    def embed_corpus(
        self,
        corpus: List[str],
        batch_size=64,
        num_workers=8,
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.get_embeddings, text_batch)
                for text_batch in self.batchify(corpus, batch_size)
            ]

            with tqdm(total=len(corpus)) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(batch_size)

            embeddings = []
            for future in futures:
                data = future.result()
                embeddings.extend(data)

            return embeddings

    def generate_embeddings(self, df, column_name):
        questions = df[column_name].astype(str).tolist()
        embeddings = self.embed_corpus(questions)
        return embeddings
