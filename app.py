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
    "lcom": 10,
    "ca": 1,
    "avg_cc": 0.75
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
    m = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc|avg_cc)", rule)
    return m.group(1) if m else None

# =========================
# FILTER RANGE RULES (NEW)
# =========================
def is_simple_rule(rule: str):
    """
    Remove rules like:
    30 < loc ≤ 109
    6 < cbo ≤ 12
    """

    # detect range pattern (two-sided inequality)
    if re.search(r"\d+(\.\d+)?\s*<\s*\w+\s*[≤<>=]\s*\d+(\.\d+)?", rule):
        return False

    # extra safety: multiple comparisons
    if rule.count("<") > 1 or rule.count(">") > 1:
        return False

    return True

# =========================
# SMART LIME
# =========================
def smart_lime(lime_exp, metrics):
    selected = {}
    for rule, _ in lime_exp:
        if not is_simple_rule(rule):
            continue

        feat = extract_feature(rule)
        if feat in THRESHOLDS and metrics.get(feat, 0) > THRESHOLDS[feat]:
            selected[feat] = rule
    return selected

# =========================
# PSEUDO ANCHOR
# =========================
def pseudo_anchor(metrics):
    anchor = {}
    for feat, thr in THRESHOLDS.items():
        if metrics.get(feat, 0) > thr:
            anchor[feat] = f"{feat} > {thr}"
    return anchor

# =========================
# MERGE
# =========================
def build_anchor_final(anchor, lime):
    final = {}

    for f, r in anchor.items():
        final[f] = r

    for f, r in lime.items():
        final[f] = r

    return final

# =========================
# INTERSECTION
# =========================
def build_intersection(anchor_final, lime):
    return {
        f: lime[f]
        for f in anchor_final.keys()
        if f in lime
    }

# =========================
# HUMAN EXPLANATION
# =========================
def humanize(intersection):
    mapping = {
        "loc": "The class is very large and hard to maintain.",
        "wmc": "High method complexity increases defect risk.",
        "rfc": "Too many method calls increase execution complexity.",
        "cbo": "High coupling reduces modularity.",
        "npm": "Too many public methods expose internal design.",
        "dit": "Deep inheritance hierarchy makes behavior unclear.",
        "lcom": "Low cohesion indicates poor class design.",
        "ca": "High coupling with other classes detected."
    }

    return [mapping[f] for f in intersection if f in mapping]

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
    # LIME (FILTERED)
    # =========================
    raw = lime_explainer.explain_instance(
        X[0], model.predict_proba, num_features=30
    ).as_list()

    lime_raw = [(r, w) for r, w in raw if is_simple_rule(r)]

    full_lime = lime_raw

    # =========================
    # SMART LIME
    # =========================
    lime_rules = smart_lime(lime_raw, metrics)

    # =========================
    # ANCHOR
    # =========================
    anchor_rules = pseudo_anchor(metrics)

    # =========================
    # FINAL ANCHOR
    # =========================
    anchor_final = build_anchor_final(anchor_rules, lime_rules)

    # =========================
    # INTERSECTION
    # =========================
    intersection = build_intersection(anchor_final, lime_rules)

    # =========================
    # OUTPUT
    # =========================
    st.metric("Defect Probability", f"{prob*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME (Clean Rules Only)")
        for r, _ in full_lime:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Dynamic)")
        for r in anchor_final.values():
            st.success(r)

    st.subheader("Intersection (Key Signal)")
    for r in intersection.values():
        st.warning(r)

    st.subheader("Expert Explanation")
    for exp in humanize(intersection):
        st.info(exp)
