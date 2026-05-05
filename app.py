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

VALID_FEATURES = {"wmc","rfc","cbo","loc","npm","dit","noc","lcom","ca","avg_cc"}

# =========================
# METRICS
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
# LIME CLEAN
# =========================
def clean_lime(lime_exp):
    return [(r,w,extract_feature(r)) for r,w in lime_exp if extract_feature(r) in VALID_FEATURES]

# =========================
# RULE TYPE
# =========================
def rule_type(rule):
    if "<" in rule and ">" in rule:
        return "range"
    if ">" in rule:
        return "gt"
    if "<" in rule:
        return "lt"
    return "other"

# =========================
# 🔥 DYNAMIC ANCHOR → RULES
# =========================
def dynamic_anchor_rules(X, metrics, model, feature_names, num_samples=200):

    base_pred = model.predict(X.reshape(1, -1))[0]
    importance = {f: 0.0 for f in feature_names}

    for _ in range(num_samples):
        noise = np.random.normal(0, 0.1, size=X.shape)
        X_pert = X + noise
        pred = model.predict(X_pert.reshape(1, -1))[0]

        if pred == base_pred:
            for i,f in enumerate(feature_names):
                importance[f] += abs(noise[i])

    for k in importance:
        importance[k] /= num_samples

    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    anchor = {}
    coverage = 0
    total = sum(v for _,v in sorted_feats)+1e-9

    for f,score in sorted_feats:

        if f not in VALID_FEATURES:
            continue

        coverage += score/total

        val = metrics.get(f, 0)
        anchor[f] = f"{f} > {val:.2f}"

        if coverage > 0.7:
            break

    return anchor

# =========================
# 🔥 MERGE (YOUR LOGIC)
# =========================
def merge_rules(lime, anchor):

    lime_map = {f:r for r,w,f in lime}
    final = dict(anchor)

    for f, lime_rule in lime_map.items():

        if f in anchor:

            t = rule_type(lime_rule)

            if t in ["gt","range"]:
                final[f] = lime_rule  # override with LIME

    return final

# =========================
# INTERSECTION
# =========================
def build_intersection(lime, anchor):
    lime_map = {f:r for r,w,f in lime}
    inter = {}

    for f in anchor:
        if f in lime_map:
            t = rule_type(lime_map[f])
            if t in ["gt","range"]:
                inter[f] = lime_map[f]
            else:
                inter[f] = anchor[f]

    return inter

# =========================
# HUMAN
# =========================
def humanize(inter):
    mapping = {
        "wmc":"High complexity in methods.",
        "rfc":"Too many method calls.",
        "cbo":"High coupling.",
        "npm":"Too many public methods.",
        "loc":"Large class size.",
        "lcom":"Low cohesion.",
        "dit":"Deep inheritance.",
        "ca":"High dependency.",
        "avg_cc":"Complex control flow."
    }
    return [mapping[f] for f in inter if f in mapping]

# =========================
# UI
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:
    code = file.read().decode("utf-8")

    metrics = extract_metrics(code)
    X = prepare(metrics)

    prob = float(model.predict_proba(X)[0][1])

    lime_raw = lime_explainer.explain_instance(
        X[0], model.predict_proba, num_features=30
    ).as_list()

    lime_clean = clean_lime(lime_raw)

    anchor = dynamic_anchor_rules(X[0], metrics, model, feature_names)

    merged = merge_rules(lime_clean, anchor)

    intersection = build_intersection(lime_clean, anchor)

    # =========================
    # OUTPUT
    # =========================
    st.metric("Defect Probability", f"{prob*100:.1f}%")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        for r,_,_ in lime_clean:
            st.write("•", r)

    with col2:
        st.subheader("Anchor (Dynamic Rules)")
        for r in merged.values():
            st.success(r)

    st.subheader("Intersection (Final Signal)")
    for r in intersection.values():
        st.warning(r)

    st.subheader("Human Explanation")
    for exp in humanize(intersection):
        st.info(exp)
