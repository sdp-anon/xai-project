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

# ✅ NEW: Google Sheets
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
# GOOGLE SHEETS CONNECTION
# =========================
def connect_to_gsheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client

def get_sheet():
    client = connect_to_gsheet()
    sheet = client.open("YOUR_SHEET_NAME").sheet1  # 🔥 change this
    return sheet

# =========================
# LOAD MODEL + DATA
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

    model_path = f"{extract_path}/nasa_model.pkl"
    feature_path = f"{extract_path}/nasa_feature_names.json"
    train_path = f"{extract_path}/nasa_X_train.csv"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found after unzip!")
        st.stop()

    model = joblib.load(model_path)

    with open(feature_path) as f:
        feature_names = json.load(f)

    X_train = pd.read_csv(train_path)
    X_train_np = X_train[feature_names].values

    lime_explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        class_names=["clean", "buggy"],
        mode="classification"
    )

    return model, feature_names, lime_explainer


model, feature_names, lime_explainer = load_resources()

# =========================
# HEURISTICS
# =========================
HEURISTIC_THRESHOLDS = {
    "wmc": 12, "rfc": 60, "cbo": 5, "loc": 100,
    "npm": 10, "dit": 3, "noc": 2, "lcom": 10
}

# =========================
# FUNCTIONS
# =========================
def extract_metrics(code: str):
    return {
        "wmc": len(re.findall(r"(public|private|protected).*?\(", code)),
        "dit": code.count("extends"),
        "noc": code.count("class "),
        "cbo": code.count("import "),
        "rfc": len(re.findall(r"\w+\(", code)),
        "lcom": code.count("this."),
        "ca": len(re.findall(r"\w+\(", code)),
        "ce": code.count("import "),
        "npm": code.count("public "),
        "loc": len(code.split("\n"))
    }

def prepare_features(metrics):
    return np.array([[metrics.get(f, 0) for f in feature_names]])

def extract_feature_name(rule_string):
    match = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule_string)
    return match.group(1) if match else None

def smart_select_lime(lime_exp, metrics):
    selected = []
    for rule, weight in lime_exp:
        if ("<" in rule and ">" in rule) or (rule.count('<') + rule.count('>') > 1):
            continue

        feature = extract_feature_name(rule)
        if feature in HEURISTIC_THRESHOLDS:
            if metrics.get(feature, 0) > HEURISTIC_THRESHOLDS[feature]:
                selected.append(rule)
    return selected

def generate_human_explanation(smart_rules):
    mapping = {
        "loc": "High LOC → possible God Class.",
        "wmc": "High WMC → too many complex methods.",
        "rfc": "High RFC → too many method calls.",
        "cbo": "High coupling → fragile class.",
        "npm": "Too many public methods.",
        "dit": "Deep inheritance → harder to understand.",
        "lcom": "Low cohesion → poor design."
    }
    return list({mapping[extract_feature_name(r)] for r in smart_rules if extract_feature_name(r) in mapping})

def predict_and_explain(code):
    metrics = extract_metrics(code)
    X = prepare_features(metrics)

    prob = float(model.predict_proba(X)[0][1])

    lime_raw = lime_explainer.explain_instance(
        X[0], model.predict_proba, num_features=10
    ).as_list()

    smart_lime = smart_select_lime(lime_raw, metrics)
    human = generate_human_explanation(smart_lime)

    return {
        "prob": prob,
        "severity": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "lime": lime_raw,
        "combined": smart_lime,
        "human": human
    }

# =========================
# UI
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:
    code = file.read().decode("utf-8")
    res = predict_and_explain(code)

    st.metric("Defect Probability", f"{res['prob']*100:.1f}%", delta=res["severity"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LIME")
        for item in res["lime"]:
            st.write(item)

    with col2:
        st.subheader("Rules")
        for rule in res["combined"]:
            st.success(rule)

    st.subheader("Expert Explanation")
    for exp in res["human"]:
        st.info(exp)

    # =========================
    # SURVEY
    # =========================
    with st.form("survey"):
        st.subheader("Evaluation")

        clarity = st.slider("Easy to understand", 1, 5)
        usefulness = st.slider("Helpful for defects", 1, 5)
        trust = st.slider("Trust level", 1, 5)
        effort = st.slider("Mental effort", 1, 5)

        preferred = st.radio("Preferred explanation", ["LIME", "Rules"])
        comments = st.text_area("Comments")

        if st.form_submit_button("Submit"):
            data = {
                "time": datetime.now().isoformat(),
                "user": st.session_state.user_id,
                "file": file.name,
                "prob": res["prob"],
                "severity": res["severity"],
                "clarity": clarity,
                "usefulness": usefulness,
                "trust": trust,
                "effort": effort,
                "preferred": preferred,
                "comments": comments
            }

            try:
                sheet = get_sheet()

                sheet.append_row([
                    data["time"],
                    data["user"],
                    data["file"],
                    data["prob"],
                    data["severity"],
                    data["clarity"],
                    data["usefulness"],
                    data["trust"],
                    data["effort"],
                    data["preferred"],
                    data["comments"]
                ])

                st.success("Saved to Google Sheets")

            except Exception as e:
                st.error(f"Error saving to Google Sheets: {e}")
