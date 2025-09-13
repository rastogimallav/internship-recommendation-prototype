# frontend/helper_ui.py
import streamlit as st

def show_match_table(resumes, internships, scores):
    """
    scores: list of arrays (each array: similarity scores vs internships)
    """
    for i, sc in enumerate(scores):
        st.markdown(f"**Resume {i+1}:**")
        ranked_idx = sc.argsort()[::-1]
        for rank, j in enumerate(ranked_idx, start=1):
            st.write(f"{rank}. Internship {j+1} — score: {sc[j]:.3f}")
        st.write("---")

def show_skill_gap(resume_text, internship_text):
    # naive skill gap: tokens present in internship but not resume
    resume_tokens = set([t.strip(",.()").lower() for t in resume_text.split()])
    internship_tokens = set([t.strip(",.()").lower() for t in internship_text.split()])
    missing = internship_tokens - resume_tokens
    # narrow to words of reasonable length
    missing = [m for m in missing if len(m) > 2]
    if missing:
        st.write("**Suggested skills / missing keywords:**", ', '.join(missing[:10]))
    else:
        st.write("No obvious missing skills detected.")

