import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import re
import os
import zipfile
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
# REMOVE RANGE RULES
# =========================
def is_simple_rule(rule):
    if re.search(r"\d+(\.\d+)?\s*<\s*\w+\s*[≤<>=]\s*\d+", rule):
        return False
    return True

# =========================
# CLEAN LIME
# =========================
def clean_lime(lime_exp):
    cleaned = []
    for rule, weight in lime_exp:
        if not is_simple_rule(rule):
            continue

        feat = extract_feature(rule)
        if feat in VALID_FEATURES:
            cleaned.append((rule, feat))
    return cleaned

# =========================
# DYNAMIC ANCHOR → RULES
# =========================
def dynamic_anchor_rules(X, model, feature_names):

    base_pred = model.predict(X.reshape(1, -1))[0]

    importance = {f: 0.0 for f in feature_names}

    for _ in range(200):
        X_pert = X.copy()
        noise = np.random.normal(0, 0.1, size=X.shape)
        X_pert = X_pert + noise

        pred = model.predict(X_pert.reshape(1, -1))[0]

        if pred == base_pred:
            for i, f in enumerate(feature_names):
                importance[f] += abs(noise[i])

    # normalize
    for k in importance:
        importance[k] /= 200

    values = np.array(list(importance.values()))
    threshold = np.percentile(values, 60)

    rules = {}

    for i, f in enumerate(feature_names):
        if f not in VALID_FEATURES:
            continue

        score = importance[f]

        if score >= threshold:
            val = X[i]

            # ❌ REMOVE weak rules
            if val <= 0:
                continue

            rules[f] = f"{f} > {round(val,2)}"

    return rules

# =========================
# MERGE (YOUR LOGIC)
# =========================
def merge_anchor_with_lime(anchor, lime):

    lime_dict = {f: r for r, f in lime}

    final_anchor = {}
    intersection = {}

    for f, rule in anchor.items():

        if f in lime_dict:

            lime_rule = lime_dict[f]

            # ✔ ONLY take LIME if it is ">"
            if ">" in lime_rule and "<" not in lime_rule:
                final_anchor[f] = lime_rule
                intersection[f] = lime_rule

        else:
            final_anchor[f] = rule

    return final_anchor, intersection

# =========================
# HUMAN EXPLANATION
# =========================
def humanize(intersection):
    mapping = {
        "loc": "The class is large → hard to maintain.",
        "wmc": "High complexity in methods.",
        "rfc": "Too many method calls.",
        "cbo": "High coupling between classes.",
        "npm": "Too many public methods exposed.",
        "dit": "Deep inheritance increases complexity.",
        "lcom": "Low cohesion in the class.",
        "ca": "High dependency on other classes.",
        "avg_cc": "Complex control flow detected."
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
        X[0], model.predict_proba, num_features=30
    ).as_list()

    lime_clean = clean_lime(lime_raw)

    # =========================
    # ANCHOR
    # =========================
    anchor_rules = dynamic_anchor_rules(X[0], model, feature_names)

    # =========================
    # MERGE + INTERSECTION
    # =========================
    anchor_final, intersection = merge_anchor_with_lime(anchor_rules, lime_clean)

    # =========================
    # OUTPUT
    # =========================
    st.metric("Defect Probability", f"{prob*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        for r, _ in lime_clean:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Final)")
        for r in anchor_final.values():
            st.success(r)

    st.subheader("Intersection (Strict Agreement)")
    for r in intersection.values():
        st.warning(r)

    st.subheader("Explanation")
    for exp in humanize(intersection):
        st.info(exp)
