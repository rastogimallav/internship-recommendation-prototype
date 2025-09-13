# ai_engine/level1_keywords.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_level1(resumes, internships):
    """
    Basic TF-IDF based keyword matching.
    resumes: list of resume texts
    internships: list of internship texts
    returns: list of score arrays per resume
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    results = []
    for r in resumes:
        # Fit on single resume + internships to get reasonable vector space
        docs = [r] + internships
        tfidf = vectorizer.fit_transform(docs)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        results.append(scores)
    return results

