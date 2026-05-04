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
# VALID FEATURES
# =========================
VALID_FEATURES = {
    "wmc", "rfc", "cbo", "loc",
    "npm", "dit", "noc", "lcom",
    "ca", "avg_cc"
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
    m = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|npm|loc|ca|avg_cc)", rule)
    return m.group(1) if m else None

# =========================
# FILTER LIME RULES
# =========================
def clean_lime(lime_exp):
    cleaned = []
    for rule, weight in lime_exp:
        feat = extract_feature(rule)
        if feat in VALID_FEATURES:
            cleaned.append((rule, weight, feat))
    return cleaned

# =========================
# DYNAMIC ANCHOR (IMPROVED)
# =========================
def dynamic_anchor(X, model, feature_names, num_samples=250, noise_level=0.12):

    base_pred = model.predict(X.reshape(1, -1))[0]
    importance = {f: 0.0 for f in feature_names}

    for _ in range(num_samples):
        X_pert = X.copy()
        noise = np.random.normal(0, noise_level, size=X.shape)
        X_pert = X_pert + noise

        pred = model.predict(X_pert.reshape(1, -1))[0]

        if pred == base_pred:
            for i, f in enumerate(feature_names):
                importance[f] += abs(noise[i])

    # normalize
    for k in importance:
        importance[k] /= num_samples

    values = np.array(list(importance.values()))

    # 🔥 LOWER = MORE FEATURES
    threshold = np.percentile(values, 55)

    anchor = {
        f: f"{f} (stability={round(score,2)})"
        for f, score in importance.items()
        if score >= threshold and f in VALID_FEATURES
    }

    return anchor

# =========================
# INTERSECTION
# =========================
def build_intersection(anchor, lime):
    lime_feats = {f for _, _, f in lime}

    return {
        f: anchor[f]
        for f in anchor
        if f in lime_feats
    }

# =========================
# HUMAN EXPLANATION
# =========================
def humanize(intersection):
    mapping = {
        "loc": "Large class size increases complexity.",
        "wmc": "High method complexity increases defect risk.",
        "rfc": "Too many method calls increase execution cost.",
        "cbo": "High coupling reduces modularity.",
        "npm": "Too many public methods expose internal logic.",
        "dit": "Deep inheritance increases complexity.",
        "lcom": "Low cohesion indicates poor design.",
        "ca": "High external dependencies detected.",
        "avg_cc": "High control flow complexity."
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
    # LIME
    # =========================
    lime_raw = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=30
    ).as_list()

    lime_clean = clean_lime(lime_raw)

    # =========================
    # ANCHOR
    # =========================
    anchor = dynamic_anchor(X[0], model, feature_names)

    # =========================
    # INTERSECTION
    # =========================
    intersection = build_intersection(anchor, lime_clean)

    # =========================
    # OUTPUT
    # =========================
    st.metric("Defect Probability", f"{prob*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME (Full)")
        for r, w, f in lime_clean:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Dynamic Model-Based)")
        for r in anchor.values():
            st.success(r)

    st.subheader("Intersection (Agreement Signal)")
    for r in intersection.values():
        st.warning(r)

    st.subheader("Explanation")
    for exp in humanize(intersection):
        st.info(exp)
