"""Medical Symptom Triage — Streamlit ML demo.

Rule-based symptom checker that suggests urgency level.
Educational tool only — NOT a medical diagnosis system.

Usage:
    streamlit run main.py
"""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Symptom Triage", layout="wide")
st.title("🏥 Medical Symptom Triage")
st.warning("⚠️ **Disclaimer:** This is an educational tool only. It does NOT provide medical advice. Always consult a qualified healthcare professional for medical concerns.")

# ── Symptom knowledge base ────────────────────────────────────────────────────

EMERGENCY_SYMPTOMS = {
    "chest pain", "chest tightness", "difficulty breathing", "shortness of breath",
    "sudden numbness", "sudden weakness", "facial drooping", "arm weakness",
    "speech difficulty", "slurred speech", "severe headache", "sudden severe headache",
    "loss of consciousness", "unresponsive", "seizure", "severe allergic reaction",
    "anaphylaxis", "coughing blood", "vomiting blood", "severe abdominal pain",
    "signs of stroke", "severe chest pressure",
}

URGENT_SYMPTOMS = {
    "high fever", "fever over 39", "persistent vomiting", "severe dehydration",
    "head injury", "broken bone", "deep cut", "severe pain", "eye injury",
    "severe burn", "urinary infection", "kidney pain", "appendix pain",
    "difficulty swallowing", "severe dizziness", "fainting", "heart palpitations",
    "rapid heartbeat", "panic attack", "moderate chest pain",
}

ROUTINE_SYMPTOMS = {
    "mild fever", "cold", "cough", "sore throat", "runny nose", "mild headache",
    "muscle ache", "fatigue", "minor cut", "minor bruise", "mild nausea",
    "stomach ache", "constipation", "diarrhea", "mild rash", "back pain",
    "ear ache", "dental pain", "eye strain", "bloating", "indigestion",
    "mild anxiety", "insomnia", "dry skin", "acne", "minor sprain",
}

CONDITIONS_MAP = {
    "chest pain":           ["Heart attack", "Angina", "Pulmonary embolism", "GERD"],
    "shortness of breath":  ["Asthma", "Pneumonia", "Heart failure", "Anxiety"],
    "severe headache":      ["Migraine", "Hypertension", "Meningitis"],
    "high fever":           ["Flu", "COVID-19", "Bacterial infection"],
    "sore throat":          ["Strep throat", "Viral pharyngitis", "Cold"],
    "stomach ache":         ["Gastritis", "IBS", "Appendicitis (if severe+right side)"],
    "cough":                ["Cold", "Bronchitis", "Asthma", "COVID-19"],
    "dizziness":            ["Vertigo", "Dehydration", "Low blood pressure"],
    "rash":                 ["Eczema", "Allergic reaction", "Contact dermatitis"],
    "back pain":            ["Muscle strain", "Disc herniation", "Kidney stone"],
}

ALL_SYMPTOMS = sorted(
    EMERGENCY_SYMPTOMS | URGENT_SYMPTOMS | ROUTINE_SYMPTOMS
)


def triage(selected: list[str], age: int, duration_days: int,
           pre_existing: list[str]) -> dict:
    sel_lower = {s.lower() for s in selected}

    emergency = sel_lower & EMERGENCY_SYMPTOMS
    urgent    = sel_lower & URGENT_SYMPTOMS
    routine   = sel_lower & ROUTINE_SYMPTOMS

    # Age adjustments
    age_multiplier = 1.5 if age > 65 or age < 5 else 1.0
    # Duration adjustments
    dur_boost = 1 if duration_days > 7 else 0

    em_score  = len(emergency) * 3 * age_multiplier
    urg_score = len(urgent)    * 2 * age_multiplier + dur_boost
    rou_score = len(routine)

    if em_score  > 0: level, color = "🚨 EMERGENCY",       "red"
    elif urg_score > 0: level, color = "⚠️ URGENT",         "orange"
    elif rou_score > 0: level, color = "🟡 ROUTINE",        "yellow"
    else:               level, color = "✅ MONITOR AT HOME", "green"

    conds = []
    for s in selected:
        conds.extend(CONDITIONS_MAP.get(s.lower(), []))

    return {
        "level":          level,
        "emergency_syms": list(emergency),
        "urgent_syms":    list(urgent),
        "conditions":     list(set(conds)),
        "em_score":       em_score,
        "urg_score":      urg_score,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Symptom Library", "About"])

with tab1:
    st.subheader("Patient Information")
    c1, c2 = st.columns(2)
    age      = c1.slider("Patient Age", 0, 100, 35)
    duration = c2.slider("Duration of symptoms (days)", 0, 30, 1)

    pre_existing = st.multiselect(
        "Pre-existing conditions",
        ["Diabetes","Hypertension","Heart disease","Asthma","COPD","Immunocompromised","Pregnancy"]
    )

    st.subheader("Select Symptoms")
    selected_syms = st.multiselect(
        "Current symptoms",
        options=sorted(ALL_SYMPTOMS),
        help="Select all symptoms the patient is experiencing"
    )

    custom = st.text_input("Other symptoms (comma-separated)")
    if custom:
        selected_syms += [s.strip() for s in custom.split(",") if s.strip()]

    if st.button("🔍 Assess Triage Level", type="primary") or selected_syms:
        if not selected_syms:
            st.info("Please select at least one symptom.")
        else:
            result = triage(selected_syms, age, duration, pre_existing)
            st.divider()
            st.subheader(f"Triage Result: {result['level']}")

            if result["emergency_syms"]:
                st.error(f"🚨 EMERGENCY symptoms detected: {', '.join(result['emergency_syms'])}\n\n**Call 911 or go to the nearest emergency room immediately.**")
            elif result["urgent_syms"]:
                st.warning(f"⚠️ Urgent symptoms: {', '.join(result['urgent_syms'])}\n\nSeek medical attention within 24 hours.")
            else:
                st.success("Symptoms appear routine. Monitor at home and see a doctor if symptoms worsen or persist.")

            if result["conditions"]:
                st.subheader("Possible Conditions (Educational Only)")
                for c in result["conditions"][:8]:
                    st.markdown(f"• {c}")
                st.caption("This list is not a diagnosis. Many conditions share symptoms.")

            if age > 65 or age < 5:
                st.info("⚠️ Patient is in a higher-risk age group. Err on the side of caution.")
            if pre_existing:
                st.info(f"⚠️ Pre-existing conditions noted: {', '.join(pre_existing)}. These may affect severity.")

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🚨 Emergency")
        for s in sorted(EMERGENCY_SYMPTOMS):
            st.markdown(f"• {s}")
    with col2:
        st.subheader("⚠️ Urgent")
        for s in sorted(URGENT_SYMPTOMS):
            st.markdown(f"• {s}")
    with col3:
        st.subheader("🟡 Routine")
        for s in sorted(ROUTINE_SYMPTOMS):
            st.markdown(f"• {s}")

with tab3:
    st.markdown("""
    ### How Triage Works
    This tool classifies symptoms into 4 urgency levels:
    - **🚨 Emergency** — Call 911 or go to ER immediately
    - **⚠️ Urgent** — See a doctor within 24 hours
    - **🟡 Routine** — Schedule an appointment
    - **✅ Monitor** — Rest and monitor symptoms

    ### Key factors considered
    - Symptom severity (Emergency > Urgent > Routine)
    - Patient age (very young < 5, elderly > 65 = higher risk)
    - Duration of symptoms
    - Pre-existing conditions

    ### Important Limitations
    - This is purely rule-based, not ML-powered
    - Cannot account for symptom combinations and severity nuances
    - Should never replace professional medical evaluation
    """)
