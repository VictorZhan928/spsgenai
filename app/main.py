# from fastapi import FastAPI
# import numpy as np
# from typing import List

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello, FastAPI with UV!"}

# from typing import Union
# from fastapi import FastAPI
# from pydantic import BaseModel
# from app.bigram_model import BigramModel

# app = FastAPI()

# # Sample corpus for the bigram model
# corpus = [
#     "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
# It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
#     "this is another example sentence",
#     "we are generating text based on bigram probabilities",
#     "bigram models are simple but effective"
# ]

# bigram_model = BigramModel(corpus)

# class TextGenerationRequest(BaseModel):
#     start_word: str
#     length: int

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post("/generate")
# def generate_text(request: TextGenerationRequest):
#     generated_text = bigram_model.generate_text(request.start_word, request.length)
#     return {"generated_text": generated_text}

# @app.get("/gaussian/")
# def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> List[float]:
#     """Sample from a Gaussian distribution with given mean and variance."""
#     std_dev = np.sqrt(variance)
#     sample = np.random.normal(mean, std_dev, size)
#     return sample.tolist()

# main.py
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

import numpy as np

# Your bigram model
from app.bigram_model import BigramModel

# ---------- FastAPI app ----------
app = FastAPI(title="Module 1+2 API", version="1.0.0")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI is running. See /docs for endpoints."}

# ---------- Bigram endpoints (Module 1 part you already had) ----------
# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 20

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    if request.length < 1:
        raise HTTPException(status_code=400, detail="length must be >= 1")
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.get("/gaussian/")
def sample_gaussian(mean: float = 0.0, variance: float = 1.0, size: int = 1) -> List[float]:
    """Sample from a Gaussian distribution with given mean and variance."""
    if variance < 0:
        raise HTTPException(status_code=400, detail="variance must be >= 0")
    std_dev = float(np.sqrt(variance))
    sample = np.random.normal(float(mean), std_dev, int(size))
    return sample.tolist()

# ---------- spaCy embeddings (Module 2 requirement) ----------
# We try to load a large model and gracefully fall back if it's missing.
import spacy

def _load_spacy_model():
    # Try large, then medium, then small
    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(name), name
        except Exception:
            continue
    raise RuntimeError(
        "No spaCy English model found. Install one of: "
        "`python -m spacy download en_core_web_lg` (preferred) "
        "or `en_core_web_md` or `en_core_web_sm`."
    )

_nlp, _nlp_name = _load_spacy_model()

class EmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    """
    Return the embedding vector for a single token using the loaded spaCy model.
    Note: small models (en_core_web_sm) do not include word vectors, only context-sensitive embeddings.
    """
    doc = _nlp(request.word.strip())
    if len(doc) == 0:
        raise HTTPException(status_code=400, detail="No tokens found in input word.")
    token = doc[0]
    vec = token.vector
    return {
        "model": _nlp_name,
        "word": token.text,
        "dim": int(vec.shape[0]),
        "embedding": vec.tolist(),
    }

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    """
    Compute cosine similarity between two words via spaCy.
    """
    doc1 = _nlp(request.word1.strip())
    doc2 = _nlp(request.word2.strip())
    if len(doc1) == 0 or len(doc2) == 0:
        raise HTTPException(status_code=400, detail="Inputs must contain at least one token each.")
    sim = doc1.similarity(doc2)
    return {
        "model": _nlp_name,
        "word1": doc1.text,
        "word2": doc2.text,
        "similarity": float(sim),
    }

