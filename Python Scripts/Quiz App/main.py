"""Quiz App — Streamlit app.

Create multiple-choice quizzes from a JSON file or enter questions
manually.  Tracks score, shows explanations, and supports timed mode.

Usage:
    streamlit run main.py
"""

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Quiz App", layout="centered")
st.title("🧠 Quiz App")

# ---------------------------------------------------------------------------
# Default sample quiz
# ---------------------------------------------------------------------------

SAMPLE_QUIZ = [
    {
        "question": "What is the capital of France?",
        "options": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": "Paris",
        "explanation": "Paris is the capital and largest city of France.",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "options": ["Venus", "Mercury", "Earth", "Mars"],
        "answer": "Mercury",
        "explanation": "Mercury is the first planet from the Sun.",
    },
    {
        "question": "What is 12 × 12?",
        "options": ["132", "144", "124", "154"],
        "answer": "144",
        "explanation": "12 × 12 = 144.",
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "options": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"],
        "answer": "William Shakespeare",
        "explanation": "Romeo and Juliet was written by William Shakespeare around 1594–1596.",
    },
    {
        "question": "What is the chemical symbol for water?",
        "options": ["H2O", "CO2", "O2", "NaCl"],
        "answer": "H2O",
        "explanation": "Water is composed of two hydrogen atoms and one oxygen atom: H₂O.",
    },
]


# ---------------------------------------------------------------------------
# Load quiz
# ---------------------------------------------------------------------------
source = st.radio("Quiz source", ["Use sample quiz", "Upload JSON file"], horizontal=True)

questions = SAMPLE_QUIZ
if source == "Upload JSON file":
    uploaded = st.file_uploader("Upload quiz JSON", type="json")
    if uploaded:
        try:
            questions = json.load(uploaded)
            st.success(f"Loaded {len(questions)} questions.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            questions = SAMPLE_QUIZ

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "q_idx" not in st.session_state:
    st.session_state.q_idx = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# ---------------------------------------------------------------------------
# Start screen
# ---------------------------------------------------------------------------
if not st.session_state.quiz_started:
    st.write(f"**Questions:** {len(questions)}")
    if st.button("🚀 Start Quiz"):
        st.session_state.quiz_started = True
        st.session_state.q_idx   = 0
        st.session_state.score   = 0
        st.session_state.answers = {}
        st.session_state.start_time = time.time()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Quiz finished
# ---------------------------------------------------------------------------
if st.session_state.q_idx >= len(questions):
    elapsed = int(time.time() - st.session_state.start_time)
    st.balloons()
    st.subheader("🎉 Quiz Complete!")
    st.metric("Score", f"{st.session_state.score} / {len(questions)}")
    st.metric("Time", f"{elapsed // 60}m {elapsed % 60}s")

    st.divider()
    st.subheader("Review Answers")
    for i, q in enumerate(questions):
        user_ans = st.session_state.answers.get(i, "No answer")
        correct  = q["answer"]
        icon     = "✅" if user_ans == correct else "❌"
        with st.expander(f"{icon} Q{i+1}: {q['question']}"):
            st.write(f"**Your answer:** {user_ans}")
            st.write(f"**Correct answer:** {correct}")
            if "explanation" in q:
                st.info(q["explanation"])

    if st.button("🔄 Restart"):
        st.session_state.quiz_started = False
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Question screen
# ---------------------------------------------------------------------------
q_idx = st.session_state.q_idx
q     = questions[q_idx]

st.progress((q_idx) / len(questions))
st.write(f"**Question {q_idx + 1} of {len(questions)}**")
st.subheader(q["question"])

chosen = st.radio("Choose an answer:", q["options"], key=f"radio_{q_idx}")

if st.button("Submit Answer"):
    st.session_state.answers[q_idx] = chosen
    if chosen == q["answer"]:
        st.session_state.score += 1
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Wrong. Correct: **{q['answer']}**")
    if "explanation" in q:
        st.info(q["explanation"])

    if st.button("Next →"):
        st.session_state.q_idx += 1
        st.rerun()
