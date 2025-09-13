# ai_engine/level3_semantics.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Try to load model; heavy but more accurate
# If model cannot be loaded because missing packages, raise to let caller fallback
MODEL = None
def get_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # compact and fast
    return MODEL

def run_level3(resumes, internships):
    """
    Sentence embedding based semantic matching.
    returns: list of score arrays per resume
    """
    model = get_model()
    emb_res = model.encode(resumes, convert_to_tensor=True)
    emb_jobs = model.encode(internships, convert_to_tensor=True)
    results = []
    for r in emb_res:
        scores = util.pytorch_cos_sim(r, emb_jobs).cpu().numpy().flatten()
        results.append(scores)
    return results

