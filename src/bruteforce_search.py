import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def brute_force_search(embeddings, query_embedding, top_k=5):
    start = time.perf_counter()
    scores = cosine_similarity([query_embedding], embeddings)[0]
    latency = (time.perf_counter() - start) * 1000

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return latency, top_indices

