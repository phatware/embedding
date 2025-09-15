import numpy as np
import openai
import os
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

embed_model = None  # type: ignore

def _isPath(model_name: str) -> bool:
    """Check if the model name is a local path."""
    model_name = str(model_name)
    return model_name.startswith("/") or model_name.startswith("./") or model_name.startswith("../")

def _get_embeddings_sentence_transformer(model_name: str, texts: List[str]) -> List[np.ndarray]:
    """Get embeddings for multiple texts using SentenceTransformer."""
    global embed_model
    if embed_model is None: # or embed_model.__class__.__name__ != "SentenceTransformer":
        embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    return [np.array(e) for e in embeddings]

def get_embedding(model_name: str, text: str, model_type: str = "openai") -> np.ndarray:
    try:
        if _isPath(model_name):
            model_type = "local"
        if model_type == "local":
            return _get_embeddings_sentence_transformer(model_name, [text])[0]
        elif model_type == "openai":
            resp = openai.embeddings.create(model=model_name, input=text)
            return np.array(resp.data[0].embedding, dtype=float)
    except Exception as e:
        print(f"[embed] Error: {e}")
        return np.array([])
