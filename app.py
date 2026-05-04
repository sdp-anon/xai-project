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
# THRESHOLDS
# =========================
THRESHOLDS = {
    "wmc": 12,
    "rfc": 60,
    "cbo": 5,
    "loc": 100,
    "npm": 10,
    "dit": 3,
    "noc": 2,
    "lcom3": 0.64,
    "cbm": 0.0,
    "amc": 5.60,
    "ca": 1,
    "avg_cc": 0.75,
    "mfa": 0.0
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
        "lcom3": code.count("this."),
        "cbm": code.count("this."),
        "amc": len(code.split()) / max(len(code.split("\n")), 1),
        "npm": code.count("public "),
        "loc": len(code.split("\n")),
        "ca": len(re.findall(r"\w+\(", code)),
        "avg_cc": len(re.findall(r"if|for|while|switch", code)),
        "mfa": code.count(".")
    }

def prepare(metrics):
    return np.array([[metrics.get(f, 0) for f in feature_names]])

def extract_feature(rule):
    m = re.search(r"(wmc|npm|loc|cbo|lcom3|cbm|amc|ca|avg_cc|noc|mfa|rfc)", rule)
    return m.group(1) if m else None

# =========================
# LIME FILTER
# =========================
def smart_lime_filter(lime_exp, metrics):
    selected = []
    for rule, _ in lime_exp:
        if ("<" in rule and ">" in rule) or (rule.count('<') + rule.count('>') > 1):
            continue

        feat = extract_feature(rule)
        if feat in THRESHOLDS and metrics.get(feat, 0) > THRESHOLDS[feat]:
            selected.append(rule)
    return selected

# =========================
# PARSING RULES
# =========================
def parse_rule(rule):
    match = re.search(r"(wmc|npm|loc|cbo|lcom3|cbm|amc|ca|avg_cc|noc|mfa|rfc)", rule)
    if not match:
        return None

    feature = match.group(1)

    value_match = re.search(r"([0-9]+\.?[0-9]*)", rule)
    value = float(value_match.group(1)) if value_match else None

    return feature, value

# =========================
# NORMALIZATION
# =========================
def normalize_lime(lime_rules):
    lime_map = {}

    for r, _ in lime_rules:
        parsed = parse_rule(r)
        if not parsed:
            continue

        f, v = parsed
        if v is None:
            continue

        lime_map[f] = min(lime_map.get(f, v), v)

    return lime_map

def normalize_anchor(anchor_rules):
    anchor_map = {}

    for r in anchor_rules:
        parsed = parse_rule(r)
        if not parsed:
            continue

        f, v = parsed
        if v is not None:
            anchor_map[f] = v

    return anchor_map

# =========================
# FUSION LOGIC
# =========================
def fuse_rules(lime_rules, anchor_rules):

    lime_map = normalize_lime(lime_rules)
    anchor_map = normalize_anchor(anchor_rules)

    final_anchor = {}
    intersection = []

    all_features = set(lime_map.keys()).union(anchor_map.keys())

    for f in all_features:

        if f in lime_map and f in anchor_map:
            final_anchor[f] = min(lime_map[f], anchor_map[f])
            intersection.append(f)

        elif f in anchor_map:
            final_anchor[f] = anchor_map[f]

        elif f in lime_map:
            final_anchor[f] = lime_map[f]

    return final_anchor, intersection

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
        "lcom3": "Low cohesion.",
        "cbm": "Low modularity."
    }

    return list(set(mapping.get(extract_feature(r), "") for r in rules))

# =========================
# UI
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:

    code = file.read().decode("utf-8")

    metrics = extract_metrics(code)
    X = prepare(metrics)

    prob = float(model.predict_proba(X)[0][1])

    # =========================
    # LIME
    # =========================
    lime_raw = lime_explainer.explain_instance(
        X[0], model.predict_proba, num_features=10
    ).as_list()

    smart_lime = smart_lime_filter(lime_raw, metrics)

    # =========================
    # MODEL-DRIVEN ANCHOR (SIMULATED)
    # =========================
    anchor_rules = []
    for f, v in metrics.items():
        if f in THRESHOLDS and v > THRESHOLDS[f]:
            anchor_rules.append(f"{f} > {THRESHOLDS[f]}")

    # =========================
    # FUSION
    # =========================
    final_anchor_map, intersection_features = fuse_rules(lime_raw, anchor_rules)

    final_anchor_rules = [
        f"{f} > {v:.2f}" for f, v in final_anchor_map.items()
    ]

    intersection_rules = intersection_features

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
        st.subheader("Anchor (Dynamic) + Smart LIME")

        st.write("### Anchor (Final)")
        for r in final_anchor_rules:
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

        preferred = st.radio("Preferred", ["LIME", "Anchor", "Both"])
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
