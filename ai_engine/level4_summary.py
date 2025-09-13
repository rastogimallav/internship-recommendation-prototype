# ai_engine/level4_summary.py
from transformers import pipeline
from ai_engine.level3_semantics import run_level3
import logging

# Attempt to create a summarizer pipeline. May be heavy.
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception as e:
    logging.warning("Summarizer pipeline unavailable; falling back to extractive summarizer.")
    summarizer = None

def simple_extractive_summary(text, max_sent=2):
    # very small fallback: return first few sentences
    sents = [s.strip() for s in text.split('.') if s.strip()]
    return '. '.join(sents[:max_sent])

def summarize_resume(text):
    if summarizer is not None:
        try:
            out = summarizer(text, max_length=60, min_length=20, do_sample=False)
            return out[0]['summary_text']
        except Exception:
            return simple_extractive_summary(text)
    else:
        return simple_extractive_summary(text)

def run_level4(resumes, internships):
    # summarize resumes then use level3 semantics to match
    summarized = [summarize_resume(r) for r in resumes]
    return run_level3(summarized, internships)

