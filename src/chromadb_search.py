import time
import chromadb


def setup_chromadb(embeddings, documents, batch_size=1000):
    """
    Create ChromaDB collection and add documents in batches
    to avoid max batch size errors
    """
    client = chromadb.Client()

    collection = client.create_collection(
        name="chromadb_collection",
        metadata={"hnsw:space": "cosine"}
    )

    total_docs = len(documents)

    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)

        batch_ids = [str(i) for i in range(start_idx, end_idx)]
        batch_docs = documents[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx].tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings
        )

    return collection


def chromadb_query(collection, query_embedding, top_k=5):
    start = time.perf_counter()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    latency = (time.perf_counter() - start) * 1000
    return latency, results

