from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
import json
import re
import os
import zipfile

from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular

from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# FASTAPI INIT
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CONFIGURATION
# =====================================================
HEURISTIC_THRESHOLDS = {
    "wmc": 12, "rfc": 60, "cbo": 5, "loc": 100,
    "npm": 10, "dit": 3, "noc": 2, "lcom": 10
}

VALID_FEATURES = {
    "wmc", "rfc", "cbo", "loc",
    "npm", "dit", "noc", "lcom", "ca", "ce"
}

# =====================================================
# UNZIP MODEL (IMPORTANT FOR RENDER)
# =====================================================
ZIP_PATH = "nasa_model.zip"
MODEL_PATH = "nasa_model.pkl"

if os.path.exists(ZIP_PATH) and not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")

# =====================================================
# LOAD MODEL + DATA
# =====================================================
model = joblib.load(MODEL_PATH)

with open("nasa_feature_names.json") as f:
    feature_names = json.load(f)

# ⚠️ reduce size for deployment stability
X_train = pd.read_csv("nasa_X_train.csv").sample(800, random_state=42)
X_train_np = X_train[feature_names].values

# =====================================================
# UTILITIES
# =====================================================
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

def extract_feature(rule):
    m = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule)
    return m.group(1) if m else None

def is_simple_rule(rule):
    # remove range rules like 30 < loc ≤ 100
    if ("<" in rule and ">" in rule) or (rule.count("<") + rule.count(">") > 1):
        return False
    return True

# =====================================================
# EXPLAINERS
# =====================================================
lime_explainer = LimeTabularExplainer(
    X_train_np,
    feature_names=feature_names,
    class_names=["clean", "buggy"],
    mode="classification"
)

anchor_explainer = AnchorTabular(
    predictor=model.predict,
    feature_names=feature_names
)

anchor_explainer.fit(X_train_np)

# =====================================================
# SMART LIME
# =====================================================
def smart_lime(lime_exp, metrics):
    selected = {}

    for rule, _ in lime_exp:
        if not is_simple_rule(rule):
            continue

        feat = extract_feature(rule)

        if feat in HEURISTIC_THRESHOLDS:
            if metrics.get(feat, 0) > HEURISTIC_THRESHOLDS[feat]:
                selected[feat] = rule

    return selected

# =====================================================
# ANCHOR → CLEAN + FORCE RULE FORMAT
# =====================================================
def clean_anchor(anchor_exp):
    if not anchor_exp.anchor:
        return []

    cleaned = []

    for r in anchor_exp.anchor:
        # normalize rule format
        r = r.replace("=", ">")
        cleaned.append(r)

    return cleaned

# =====================================================
# MERGE LOGIC (YOUR RULE)
# =====================================================
def merge_anchor_lime(anchor_rules, lime_rules):
    final = {}

    # add anchor first
    for r in anchor_rules:
        feat = extract_feature(r)
        if feat:
            final[feat] = r

    # override with LIME if exists AND rule is ">"
    for feat, rule in lime_rules.items():
        if ">" in rule:
            final[feat] = rule

    return final

# =====================================================
# INTERSECTION (STRICT)
# =====================================================
def build_intersection(anchor_final, lime_rules):
    return {
        f: lime_rules[f]
        for f in anchor_final
        if f in lime_rules
    }

# =====================================================
# HUMAN EXPLANATION
# =====================================================
def humanize(features):
    mapping = {
        "loc": "Large class size increases maintenance difficulty.",
        "wmc": "High complexity increases defect risk.",
        "rfc": "Too many method calls increase execution complexity.",
        "cbo": "High coupling reduces modularity.",
        "npm": "Too many public methods expose internal design.",
        "dit": "Deep inheritance complicates behavior.",
        "lcom": "Low cohesion indicates poor design.",
        "ca": "High dependency on external classes.",
        "ce": "Excessive outgoing dependencies."
    }

    return [mapping[f] for f in features if f in mapping]

# =====================================================
# API ENDPOINT
# =====================================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    code = (await file.read()).decode("utf-8")

    metrics = extract_metrics(code)
    X = prepare_features(metrics)

    prob = float(model.predict_proba(X)[0][1])

    # =========================
    # LIME
    # =========================
    lime_raw = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=20
    ).as_list()

    full_lime = [r for r, _ in lime_raw if is_simple_rule(r)]
    lime_selected = smart_lime(lime_raw, metrics)

    # =========================
    # ANCHOR (SAFE EXECUTION)
    # =========================
    try:
        anchor_exp = anchor_explainer.explain(
            X[0],
            threshold=0.6,
            beam_size=6,
            max_anchor_size=5
        )
        anchor_rules = clean_anchor(anchor_exp)

    except Exception:
        anchor_rules = []

    # =========================
    # MERGE
    # =========================
    anchor_final = merge_anchor_lime(anchor_rules, lime_selected)

    intersection = build_intersection(anchor_final, lime_selected)

    # =========================
    # RESPONSE
    # =========================
    return {
        "defect_probability": round(prob, 3),
        "severity": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",

        "lime_full": full_lime,

        "anchor_final": list(anchor_final.values()),

        "intersection_rules": list(intersection.values()),

        "human_explanation": humanize(intersection.keys())
    }
