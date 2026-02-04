import random
import pandas as pd
from src.data_loader import prepare_documents
from src.config import *
from src.embeddings import *
from src.bm25_search import *
from src.bruteforce_search import *
from src.chromadb_search import *
from src.pinecone_search import *


documents = prepare_documents(PDF_DIR)
queries = random.sample(documents, NUM_TEST_QUERIES)

results = []

# BM25
bm25 = bm25_setup(documents)
bm25_times = []

for q in queries:
    latency, _ = bm25_search(bm25, q)
    bm25_times.append(latency)

results.append(["BM25", "N/A", sum(bm25_times)/len(bm25_times), "Medium"])

# MiniLM Embeddings
doc_embeddings = embed_minilm(documents)
query_embeddings = embed_minilm(queries)

# Brute Force
bf_times = []
for qe in query_embeddings:
    latency, _ = brute_force_search(doc_embeddings, qe)
    bf_times.append(latency)

results.append(["Brute Force", "MiniLM", sum(bf_times)/len(bf_times), "High"])

# ChromaDB HNSW
collection = setup_chromadb(doc_embeddings, documents)
chroma_times = []

for qe in query_embeddings:
    latency, _ = chromadb_query(collection, qe)
    chroma_times.append(latency)

results.append(["HNSW (ChromaDB)", "MiniLM", sum(chroma_times)/len(chroma_times), "High"])

# Pinecone HNSW
index = setup_pinecone(
    PINECONE_INDEX_NAME,
    doc_embeddings,
    documents,
    doc_embeddings.shape[1]
)

pinecone_times = []
for qe in query_embeddings:
    latency, _ = pinecone_query(index, qe)
    pinecone_times.append(latency)

results.append(["HNSW (Pinecone)", "MiniLM", sum(pinecone_times)/len(pinecone_times), "Very High"])

df = pd.DataFrame(
    results,
    columns=["Configuration", "Embedding Model", "Avg Latency (ms)", "Accuracy Level"]
)

print(df)
df.to_csv("results/benchmark_summary.csv", index=False)
