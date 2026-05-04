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
# THRESHOLDS
# =========================
THRESHOLDS = {
    "wmc": 12, "rfc": 60, "cbo": 5, "loc": 100,
    "npm": 10, "dit": 3, "noc": 2, "lcom": 10
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
    m = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|npm|loc)", rule)
    return m.group(1) if m else None

# =========================
# CLEAN RULE NORMALIZATION (NEW FIX)
# =========================
def normalize(rule):
    rule = rule.replace(" ", "")
    rule = re.sub(r"(\d+)\.0+", r"\1", rule)
    return rule

def extract_value(rule):
    nums = re.findall(r"(\d+\.?\d*)", rule)
    return float(nums[0]) if nums else None

# =========================
# SMART LIME
# =========================
def smart_lime_filter(lime_exp, metrics):
    selected = []
    for rule, _ in lime_exp:

        if ("<" in rule and ">" in rule) or (rule.count('<') + rule.count('>') > 1):
            continue

        feat = extract_feature(rule)
        if feat in THRESHOLDS and metrics.get(feat, 0) > THRESHOLDS[feat]:
            selected.append(normalize(rule))

    # 🔥 KEEP ONLY 1 RULE PER FEATURE
    best = {}
    for r in selected:
        f = extract_feature(r)
        if f and f not in best:
            best[f] = r

    return list(best.values())

# =========================
# PSEUDO ANCHOR (CLEANED)
# =========================
def pseudo_anchor(metrics):
    anchors = {}

    for feat, th in THRESHOLDS.items():
        if metrics.get(feat, 0) > th:
            anchors[feat] = f"{feat} > {th}"

    return list(anchors.values()) if anchors else ["No strong anchor conditions"]

# =========================
# INTERSECTION (FIXED LOGIC)
# =========================
def intersection(anchor, lime):
    anchor_feats = {extract_feature(r) for r in anchor}
    lime_feats = {extract_feature(r) for r in lime}

    common = anchor_feats.intersection(lime_feats)

    return [
        r for r in anchor + lime
        if extract_feature(r) in common
    ]

# =========================
# HUMAN EXPLANATION
# =========================
def humanize(rules):
    mapping = {
        "loc": "High LOC → God Class.",
        "wmc": "High complexity.",
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

    # LIME
    lime_raw = lime_explainer.explain_instance(
        X[0], model.predict_proba, num_features=10
    ).as_list()

    smart_lime = smart_lime_filter(lime_raw, metrics)

    # ANCHOR (PSEUDO)
    anchor_rules = pseudo_anchor(metrics)

    # COMBINATIONS
    union_rules = list(set(anchor_rules + smart_lime))
    intersection_rules = intersection(anchor_rules, smart_lime)

    # =========================
    # OUTPUT
    # =========================
    st.metric("Defect Probability", f"{prob*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        for r, _ in lime_raw:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Pseudo) + Smart LIME")

        st.write("### Union")
        for r in union_rules:
            st.success(r)

        st.write("### Intersection (Key Signal)")
        for r in intersection_rules:
            st.warning(r)

    st.subheader("Expert Explanation")
    for exp in humanize(smart_lime):
        st.info(exp)

    # =========================
    # SURVEY + SAVE
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
                st.error(f"Error saving: {e}")
