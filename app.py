# app.py
import streamlit as st
import os
from ai_engine import level1_keywords, level2_synonyms
from ai_engine import level3_semantics, level4_summary
from frontend.helper_ui import show_match_table, show_skill_gap

st.set_page_config(page_title="AI Internship Recommender", layout="wide")

st.title("AI-Based Internship Recommendation (Prototype)")

# --- Load example data from data/ folder
DATA_DIR = "data"
resume_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().startswith("resume")])
intern_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().startswith("internship")])

resumes = []
for f in resume_files:
    with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as fh:
        resumes.append(fh.read())

internships = []
for f in intern_files:
    with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as fh:
        internships.append(fh.read())

st.sidebar.header("Demo Controls")
level = st.sidebar.selectbox("Select AI Level", [
    "Level 1: Keyword Matching (TF-IDF)",
    "Level 2: Synonym Expansion + TF-IDF",
    "Level 3: Sentence Semantic Matching",
    "Level 4: Resume Summarization + Semantic Matching"
])
run_demo = st.sidebar.button("Run Demo")
show_raw = st.sidebar.checkbox("Show raw data (resumes & internships)")
show_explain = st.sidebar.checkbox("Show explainability")

if show_raw:
    st.subheader("Resumes (loaded)")
    for i, r in enumerate(resumes, start=1):
        st.markdown(f"**Resume {i}:**")
        st.text(r)
    st.subheader("Internships (loaded)")
    for i, j in enumerate(internships, start=1):
        st.markdown(f"**Internship {i}:**")
        st.text(j)

if run_demo:
    st.header(f"Results — {level}")
    try:
        if level.startswith("Level 1"):
            scores = level1_keywords.run_level1(resumes, internships)
        elif level.startswith("Level 2"):
            scores = level2_synonyms.run_level2(resumes, internships)
        elif level.startswith("Level 3"):
            # may raise if model not available
            scores = level3_semantics.run_level3(resumes, internships)
        else:
            scores = level4_summary.run_level4(resumes, internships)
    except Exception as e:
        st.error(f"An error occurred running the selected AI level: {e}")
        st.stop()

    # Show ranked matches
    show_match_table(resumes, internships, scores)

    # Compute simple accuracy if we assume default mapping (for demo only)
    # For demo, let's try to compute a naive accuracy if there is a 1:1 mapping by index
    import numpy as np
    ground_truth = {0:0, 1:1, 2:2} if len(resumes) >= 3 and len(internships) >= 3 else {}
    if ground_truth:
        correct = 0
        for i, sc in enumerate(scores):
            pred = int(np.argmax(sc))
            if ground_truth.get(i) == pred:
                correct += 1
        acc = correct / len(ground_truth)
        st.write(f"Demo accuracy (naive index mapping): **{acc*100:.1f}%** (for demo purposes)")

    # Explainability: show top tokens or summary-based reason
    if show_explain:
        st.header("Explainability")
        if level.startswith("Level 1") or level.startswith("Level 2"):
            st.write("Top keywords found (TF-IDF based):")
            # rough token display
            for i, r in enumerate(resumes):
                st.write(f"Resume {i+1} tokens:", ', '.join(list(set([w.strip('.,()') for w in r.split()]) )[:15]))
        elif level.startswith("Level 3"):
            st.write("Showing sentence-level similarity scores (per resume):")
            for i, sc in enumerate(scores):
                st.write(f"Resume {i+1} top match: Internship {int(sc.argmax())+1} — score {sc.max():.3f}")
        else:
            st.write("Resume summaries used:")
            # show summaries
            from ai_engine.level4_summary import summarize_resume
            for i, r in enumerate(resumes):
                st.write(f"Resume {i+1} summary:", summarize_resume(r))

    st.sidebar.header("Skill Gap Analysis")
    chosen_resume = st.sidebar.selectbox("Choose resume for skill gap", resume_files)
    chosen_intern = st.sidebar.selectbox("Choose internship", intern_files)
    with open(os.path.join(DATA_DIR, chosen_resume), 'r', encoding='utf-8') as fh:
        res_text = fh.read()
    with open(os.path.join(DATA_DIR, chosen_intern), 'r', encoding='utf-8') as fh:
        int_text = fh.read()
    show_skill_gap(res_text, int_text)

st.markdown("---")
st.caption("Prototype for SIH — shows stepwise AI proficiency levels. Replace data/ with real inputs and integrate with PM Internship portal for full system.")

