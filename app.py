import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import re
import os
import zipfile
from datetime import datetime
import uuid

from lime.lime_tabular import LimeTabularExplainer

import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="XAI Study", layout="wide")

# =========================
# USER ID
# =========================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

st.title("Software Defect Prediction Explainer")
st.caption(f"Participant ID: {st.session_state.user_id}")

# =========================
# GOOGLE SHEETS
# =========================
def connect_to_gsheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    return gspread.authorize(creds)

def get_sheet():
    return connect_to_gsheet().open("XAI Survey Results").sheet1

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_resources():
    zip_path = "nasa_model.zip"
    extract_path = "model_files"

    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    if os.path.exists(zip_path) and not os.path.exists(f"{extract_path}/nasa_model.pkl"):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    model = joblib.load(f"{extract_path}/nasa_model.pkl")

    with open(f"{extract_path}/nasa_feature_names.json") as f:
        feature_names = json.load(f)

    X_train = pd.read_csv(f"{extract_path}/nasa_X_train.csv")
    X_np = X_train[feature_names].values

    lime = LimeTabularExplainer(
        X_np,
        feature_names=feature_names,
        class_names=["clean", "buggy"],
        mode="classification"
    )

    return model, feature_names, lime

model, feature_names, lime_explainer = load_resources()

# =========================
# THRESHOLDS (SMART ANCHOR)
# =========================
THRESHOLDS = {
    "wmc": 12,
    "rfc": 60,
    "cbo": 5,
    "loc": 100,
    "npm": 10,
    "dit": 3,
    "noc": 2,
    "lcom": 10
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_metrics(code):
    return {
        "wmc": len(re.findall(r"(public|private|protected).*?\(", code)),
        "dit": code.count("extends"),
        "noc": code.count("class "),
        "cbo": code.count("import "),
        "rfc": len(re.findall(r"\w+\(", code)),
        "lcom": code.count("this."),
        "npm": code.count("public "),
        "loc": len(code.split("\n"))
    }

def prepare(metrics):
    return np.array([[metrics.get(f, 0) for f in feature_names]])

def extract_feature(rule):
    m = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule)
    return m.group(1) if m else None

# =========================
# LIME FILTER
# =========================
def smart_lime_filter(lime_exp, metrics):
    selected = []

    for rule, _ in lime_exp:
        if ("<" in rule and ">" in rule) or (rule.count("<") + rule.count(">") > 1):
            continue

        feat = extract_feature(rule)

        if feat in THRESHOLDS and metrics.get(feat, 0) > THRESHOLDS[feat]:
            selected.append(rule)

    return selected

# =========================
# MODEL-DRIVEN ANCHOR (FIXED + STABLE)
# =========================
def generate_anchor_rules(X_instance, model, feature_names, metrics):
    rules = []

    base_pred = model.predict(X_instance.reshape(1, -1))[0]

    important_feats = list(set([
        extract_feature(f"{k} > 0") for k in metrics.keys()
    ]))

    important_feats = [f for f in important_feats if f in feature_names]

    if not important_feats:
        return ["No anchor signals"]

    num_samples = 25  # optimized speed

    for feat in important_feats:
        idx = feature_names.index(feat)

        X_pert = np.repeat(X_instance.reshape(1, -1), num_samples, axis=0).astype(float)

        noise = np.random.normal(0, 0.05, size=num_samples)
        X_pert[:, idx] = X_pert[:, idx] * (1 + noise)

        preds = model.predict(X_pert)

        stability = np.mean(preds == base_pred)

        if stability > 0.80:
            rules.append(f"{feat} stable (conf={stability:.2f})")

    return rules

# =========================
# HUMAN EXPLANATION
# =========================
def humanize(rules):
    mapping = {
        "loc": "High LOC → God Class.",
        "wmc": "High complexity in methods.",
        "rfc": "Too many method calls.",
        "cbo": "High coupling.",
        "npm": "Too many public methods.",
        "dit": "Deep inheritance.",
        "lcom": "Low cohesion."
    }

    return list(set(
        mapping.get(extract_feature(r), "")
        for r in rules if extract_feature(r) in mapping
    ))

# =========================
# UI
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:

    code = file.read().decode("utf-8")

    metrics = extract_metrics(code)
    X = prepare(metrics)

    prob = float(model.predict_proba(X)[0][1])

    st.metric("Defect Probability", f"{prob*100:.1f}%")

    # =========================
    # LIME
    # =========================
    lime_raw = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=10
    ).as_list()

    smart_lime = smart_lime_filter(lime_raw, metrics)

    # =========================
    # ANCHOR (MODEL-DRIVEN)
    # =========================
    anchor_rules = generate_anchor_rules(X[0], model, feature_names, metrics)

    # =========================
    # CLEAN COMBINATION (NO DUPLICATES)
    # =========================
    union_rules = list(dict.fromkeys(anchor_rules + smart_lime))
    intersection_rules = list(set(anchor_rules) & set(smart_lime))

    # =========================
    # DISPLAY
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        for r, _ in lime_raw:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Model-Driven) + Smart LIME")

        st.write("### Anchor")
        for r in anchor_rules:
            st.success(r)

        st.write("### Smart LIME")
        for r in smart_lime:
            st.info(r)

    st.subheader("Combined Signal")

    st.write("**Union**")
    for r in union_rules:
        st.write(r)

    st.write("**Intersection (Key Signal)**")
    for r in intersection_rules:
        st.warning(r)

    st.subheader("Expert Explanation")

    for exp in humanize(smart_lime):
        st.info(exp)

    # =========================
    # SURVEY + GOOGLE SHEETS
    # =========================
    with st.form("survey"):

        clarity = st.slider("Clarity", 1, 5)
        usefulness = st.slider("Usefulness", 1, 5)
        trust = st.slider("Trust", 1, 5)
        effort = st.slider("Effort", 1, 5)

        preferred = st.radio("Preferred Explanation", ["LIME", "Anchor", "Both"])
        comments = st.text_area("Comments")

        if st.form_submit_button("Submit"):

            row = [
                datetime.now().isoformat(),
                st.session_state.user_id,
                file.name,
                prob,
                clarity,
                usefulness,
                trust,
                effort,
                preferred,
                comments
            ]

            try:
                sheet = get_sheet()
                sheet.append_row(row)
                st.success("Saved to Google Sheets ✅")
            except Exception as e:
                st.error(f"Error: {e}")
