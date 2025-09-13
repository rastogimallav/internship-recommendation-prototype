# ai_engine/level2_synonyms.py
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure WordNet is available (first run)
try:
    _ = wordnet.synsets("test")
except Exception:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Small manual skill mapping (common abbreviations / synonyms)
SKILL_MAP = {
    'js': 'javascript',
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'pandas': 'data processing',
    'sql': 'sql',
    'reactjs': 'react'
}

def expand_text_with_synonyms(text):
    tokens = [t.strip(".,()").lower() for t in text.split()]
    expanded = set(tokens)
    for t in tokens:
        if t in SKILL_MAP:
            expanded.add(SKILL_MAP[t])
        # WordNet synonyms
        for syn in wordnet.synsets(t):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return " ".join(expanded)

def run_level2(resumes, internships):
    # Expand resumes & internships with synonyms and run TF-IDF
    expanded_resumes = [expand_text_with_synonyms(r) for r in resumes]
    expanded_jobs = [expand_text_with_synonyms(j) for j in internships]
    vectorizer = TfidfVectorizer(stop_words='english')
    results = []
    for r in expanded_resumes:
        tfidf = vectorizer.fit_transform([r] + expanded_jobs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        results.append(scores)
    return results

