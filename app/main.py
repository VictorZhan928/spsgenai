# main.py
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import torch
from torchvision import transforms
from PIL import Image
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel
from app.cnn_model import SimpleCNN
import base64, io
from torchvision import transforms, utils as vutils
from app.gan_model import Generator

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

# ---------- CNN image classification (Module 2 requirement) ----------
_CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

_device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
_cnn_model = SimpleCNN(num_classes=10).to(_device)
_WEIGHTS_PATH = "artifacts/cnn_cifar10.pt"

try:
    ckpt = torch.load(_WEIGHTS_PATH, map_location=_device)
    _cnn_model.load_state_dict(ckpt["state_dict"])
    _cnn_model.eval()
    _cnn_ready = True
    print("✅ CNN weights loaded")
except Exception as e:
    _cnn_ready = False
    print("⚠️ CNN weights NOT loaded:", e)

_infer_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

class PredictResponse(BaseModel):
    top1_label: str
    top1_prob: float
    top3: list

@app.post("/classify", response_model=PredictResponse)
def classify_image(file: UploadFile = File(...)):
    if not _cnn_ready:
        raise HTTPException(status_code=503, detail="CNN weights not loaded. Train and place artifacts/cnn_cifar10.pt")
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    with torch.no_grad():
        x = _infer_tf(image).unsqueeze(0).to(_device)
        probs = torch.softmax(_cnn_model(x), dim=1).squeeze(0).cpu()
    top3_prob, top3_idx = torch.topk(probs, k=3)
    top3 = [{"label": _CIFAR10_LABELS[i], "prob": float(p)}
            for p, i in zip(top3_prob.tolist(), top3_idx.tolist())]
    return {"top1_label": top3[0]["label"], "top1_prob": top3[0]["prob"], "top3": top3}

# ---------- GAN (Module 6) ----------
_gan_ready = False
_gan_nz = 100
_gan_device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
_gan_gen = Generator(nz=_gan_nz).to(_gan_device)
_GAN_WEIGHTS_PATH = "artifacts/gan_mnist.pt"

try:
    ckpt = torch.load(_GAN_WEIGHTS_PATH, map_location=_gan_device)
    _gan_gen.load_state_dict(ckpt["state_dict"])
    _gan_nz = ckpt.get("nz", 100)
    _gan_gen.eval()
    _gan_ready = True
    print("✅ GAN generator loaded")
except Exception as e:
    print("⚠️ GAN weights NOT loaded:", e)

# --------- GAN endpoint ---------
class GANRequest(BaseModel):
    n: int = 16  # must be a perfect square (e.g., 16, 25, 36)

@app.post("/generate_gan")
def generate_gan(req: GANRequest):
    if not _gan_ready:
        raise HTTPException(
            status_code=503,
            detail="GAN weights not loaded. Train and save to artifacts/gan_mnist.pt first."
        )

    n = int(req.n)
    side = int(n ** 0.5)
    if side * side != n:
        raise HTTPException(
            status_code=400,
            detail="n must be a perfect square (16, 25, 36, ...)."
        )

    with torch.no_grad():
        z = torch.randn(n, _gan_nz, device=_gan_device)
        imgs = _gan_gen(z).cpu()  # [-1, 1]
        grid = vutils.make_grid(
            imgs, nrow=side, normalize=True, value_range=(-1, 1)
        )

    buf = io.BytesIO()
    transforms.ToPILImage()(grid).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64}
