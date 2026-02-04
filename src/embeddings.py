import os
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def embed_minilm(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True)


def embed_mpnet(texts):
    model = SentenceTransformer("all-mpnet-base-v2")
    return model.encode(texts, show_progress_bar=True)


def embed_openai(texts):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    vectors = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vectors.append(response.data[0].embedding)

    return np.array(vectors)


