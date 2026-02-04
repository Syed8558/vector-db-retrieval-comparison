import time
import numpy as np
from rank_bm25 import BM25Okapi


def bm25_setup(documents):
    """
    Create a BM25 index from a list of documents
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25


def bm25_search(bm25, query, top_k=5):
    """
    Perform BM25 search and return latency + top indices
    """
    start = time.perf_counter()
    scores = bm25.get_scores(query.lower().split())
    latency = (time.perf_counter() - start) * 1000

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return latency, top_indices
