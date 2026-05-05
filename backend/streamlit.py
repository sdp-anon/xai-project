import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import uuid

st.set_page_config(page_title="XAI Study", layout="wide")

# =========================
# Participant ID
# =========================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

st.title("Software Defect Prediction Explainer")
st.caption(f"Participant ID: {st.session_state.user_id}")

# =========================
# Upload
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:
    with st.spinner("Analyzing..."):
        try:
            res = requests.post(
                "http://localhost:8000/upload/",
                files={"file": file}
            ).json()
        except:
            st.error("Backend Error.")
            st.stop()

    # =========================
    # Prediction
    # =========================
    st.metric(
        "Defect Probability",
        f"{res['defect_probability']*100:.1f}%",
        delta=res['severity']
    )

    col1, col2 = st.columns(2)

    # =========================
    # LIME
    # =========================
    with col1:
        st.subheader("1. Local Interpretable Model-Agnostic Explanation(LIME)")

        for item in res["full_lime"]:
            st.write(f"• {item}")

    # =========================
    # Anchor + Smart LIME
    # =========================
    with col2:
        st.subheader("2. Anchor")

        if res["combined_anchor_result"]:
            for rule in res["combined_anchor_result"]:
                st.success(f"✔ {rule}")
        else:
            st.write("No strong rules detected.")

    st.divider()

    # =========================
    # Human Explanation
    # =========================
    st.subheader("Expert Reasoning")

    if res["human_explanation"]:
        for exp in res["human_explanation"]:
            st.info(exp)
    else:
        st.write("No significant issues detected.")

    st.divider()

    # =========================
    # SURVEY (RESEARCH QUALITY)
    # =========================
    with st.form("survey"):

        st.subheader("Evaluation")

        # Understanding
        st.markdown("### Understanding")
        clarity = st.slider("The explanation is easy to understand", 1, 5)
        highlight = st.slider("The explanation highlights problematic code clearly", 1, 5)

        # Usefulness
        st.markdown("### Usefulness")
        usefulness = st.slider("The explanation helps identify defects", 1, 5)
        decision = st.slider("The explanation helps me decide to refactor", 1, 5)

        # Trust
        st.markdown("### Trust")
        trust = st.slider("I trust this explanation", 1, 5)
        reliance = st.slider("I would rely on this tool in real projects", 1, 5)

        # Actionability
        st.markdown("### Actionability")
        actionable = st.slider("The explanation suggests meaningful improvements", 1, 5)
        independence = st.slider("I can act on this without other tools", 1, 5)

        # Cognitive Load (IMPORTANT)
        st.markdown("### Cognitive Effort")
        effort = st.slider("The explanation required high mental effort", 1, 5)

        # Agreement (YOUR CONTRIBUTION)
        st.markdown("### Agreement")
        agreement = st.slider("Agreement between explanations increases my confidence", 1, 5)

        # Preference
        st.markdown("### Preference")
        preferred = st.radio(
            "Which explanation style do you prefer?",
            ["LIME", "Anchor", "Both"]
        )

        comments = st.text_area("Optional comments")

        # =========================
        # SAVE
        # =========================
        if st.form_submit_button("Submit"):

            data = {
                "time": datetime.now().isoformat(),
                "participant_id": st.session_state.user_id,
                "file_name": file.name,

                # model output
                "probability": res["defect_probability"],
                "severity": res["severity"],

                # survey
                "clarity": clarity,
                "highlight": highlight,
                "usefulness": usefulness,
                "decision": decision,
                "trust": trust,
                "reliance": reliance,
                "actionable": actionable,
                "independence": independence,
                "effort": effort,
                "agreement": agreement,
                "preferred": preferred,
                "comments": comments
            }

            df = pd.DataFrame([data])

            file_path = "survey_results.csv"

            if not os.path.exists(file_path):
                df.to_csv(file_path, index=False)
            else:
                df.to_csv(file_path, mode="a", header=False, index=False)

            st.success("Response saved. Thank you!")