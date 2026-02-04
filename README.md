# Vector Database & Retrieval Algorithm Comparison

## 📌 Overview
This project presents a **comparative analysis of vector databases and retrieval algorithms** for large-scale document retrieval.  
A corpus of **1000 PDF documents** was used to evaluate different **embedding models**, **vector databases**, and **search algorithms**, focusing on **response time (latency)** and **retrieval quality**.

The study compares **traditional lexical search** with **vector-based semantic search**, highlighting trade-offs between **accuracy, speed, and scalability**.

---

## 📂 Dataset
- **Source**: 1000 PDF documents
- **Preprocessing**:
  - Text extracted from PDFs
  - Chunked into fixed-size overlapping segments
- **Query Type**: Natural language questions
- **Number of test queries**: 20
- **Top-K results**: 5
- **Similarity metric**: Cosine similarity

> ⚠️ PDFs are not included in the repository to reduce size.

---

## 🧠 Embedding Models Evaluated

| Model Name | Vector Dimension | Type |
|-----------|------------------|------|
| all-MiniLM-L6-v2 | 384 | Lightweight, fast |
| all-mpnet-base-v2 | 768 | High-quality semantic |
| text-embedding-3-small | 1536 | OpenAI cloud-based |

---

## 🗄️ Vector Databases Compared

| Database | Type | Deployment |
|--------|------|------------|
| ChromaDB | Open-source | Local |
| Pinecone | Managed cloud | Serverless |

---

## 🔍 Retrieval Algorithms

| Algorithm | Category |
|---------|----------|
| BM25 | Lexical search |
| Brute Force | Exact vector search |
| HNSW | Approximate Nearest Neighbor |

---

## ⚙️ Evaluation Configuration
- Queries: 20 natural language questions
- Top-K retrieval: 5
- Similarity metric: Cosine similarity
- Metrics evaluated:
  - Average response time (ms)
  - Qualitative retrieval accuracy

---

## 📊 Performance Results

### 🔹 Benchmark Summary

| Configuration | Embedding Model | Avg Latency (ms) | Accuracy Level |
|-------------|----------------|-----------------|----------------|
| BM25 | N/A | 210.93 | Medium |
| Brute Force | MiniLM | 21.43 | High |
| HNSW (ChromaDB) | MiniLM | 5.47 | High |
| HNSW (Pinecone) | MiniLM | 539.73 | Very High |

📁 Full results available in:  
results/benchmark_summary.csv  


---

## 🧠 Key Observations
- **BM25** is slower and less accurate for semantic queries.
- **Brute-force vector search** improves accuracy but scales poorly.
- **ChromaDB (HNSW)** provides the **lowest latency** for local workloads.
- **Pinecone** delivers **very high retrieval quality** but incurs higher latency due to network and serverless overhead.
- Both ChromaDB and Pinecone require **batched ingestion** for large datasets due to internal limits.

---

## 🏗️ Project Structure

vector-db-retrieval-comparison/
├── src/
│ ├── config.py
│ ├── data_loader.py
│ ├── embeddings.py
│ ├── bm25_search.py
│ ├── bruteforce_search.py
│ ├── chromadb_search.py
│ ├── pinecone_search.py
│ ├── evaluate.py
│ └── benchmark.py
├── results/
│ └── benchmark_summary.csv
├── experiments/
├── docs/
└── README.md


---

## ▶️ How to Run

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Set API keys (optional)
export OPENAI_API_KEY=your_key_here
export PINECONE_API_KEY=your_key_here
4️⃣ Run evaluation
python -m src.evaluate
python -m src.benchmark

🚀 Conclusion

This project demonstrates how vector-based retrieval systems outperform traditional lexical methods for semantic search tasks.
While local vector databases excel in latency-sensitive environments, managed cloud databases like Pinecone provide better scalability and retrieval quality for production-grade systems.

👤 Author

Syed Sadath G
Data Scientist

