import time
import os
from pinecone import Pinecone, ServerlessSpec


def setup_pinecone(index_name, embeddings, documents, dimension, batch_size=100):
    """
    Create Pinecone index and upsert vectors in batches
    to avoid request size limits
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)

    total_vectors = len(embeddings)

    for start in range(0, total_vectors, batch_size):
        end = min(start + batch_size, total_vectors)

        vectors = [
            (str(i), embeddings[i].tolist())
            for i in range(start, end)
        ]

        index.upsert(vectors=vectors)

    return index


def pinecone_query(index, query_embedding, top_k=5):
    start = time.perf_counter()
    result = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k
    )
    latency = (time.perf_counter() - start) * 1000
    return latency, result





