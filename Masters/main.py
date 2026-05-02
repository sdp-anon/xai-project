from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
import json
import re

from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# CONFIGURATION & HEURISTICS
# =====================================================
HEURISTIC_THRESHOLDS = {
    "wmc": 12, "rfc": 60, "cbo": 5, "loc": 100, 
    "npm": 10, "dit": 3, "noc": 2, "lcom": 10
}

# =====================================================
# LOAD MODEL + DATA
# =====================================================
model = joblib.load("nasa_model.pkl")

with open("nasa_feature_names.json") as f:
    feature_names = json.load(f)

X_train = pd.read_csv("nasa_X_train.csv")
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

def extract_feature_name(rule_string):
    match = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule_string)
    return match.group(1) if match else None

# =====================================================
# XAI LOGIC
# =====================================================
lime_explainer = LimeTabularExplainer(
    X_train_np, feature_names=feature_names,
    class_names=["clean", "buggy"], mode="classification"
)

anchor_explainer = AnchorTabular(predictor=model.predict, feature_names=feature_names)
anchor_explainer.fit(X_train_np)

def smart_select_lime(lime_exp, metrics):
    selected = []
    for rule, weight in lime_exp:
        # STRICT EXCLUSION: If it contains both < and > or more than one comparison, remove.
        if ("<" in rule and ">" in rule) or (rule.count('<') + rule.count('>') > 1):
            continue

        feature = extract_feature_name(rule)
        if feature in HEURISTIC_THRESHOLDS:
            if metrics.get(feature, 0) > HEURISTIC_THRESHOLDS[feature]:
                selected.append(rule)
    return selected

def generate_human_explanation(smart_rules):
    mapping = {
        "loc": "High Lines of Code (LOC) indicates a 'God Class' that is hard to maintain.",
        "wmc": "High Weighted Method Complexity (WMC) indicates too many logical paths.",
        "rfc": "High Response For a Class (RFC) shows excessive potential method executions.",
        "cbo": "High Coupling (CBO) indicates a fragile class with too many dependencies.",
        "npm": "High Number of Public Methods (NPM) increases the class interface surface.",
        "dit": "Deep Inheritance Tree (DIT) makes understanding behavior complex.",
        "lcom": "High LCOM suggests the class lacks a single cohesive responsibility."
    }
    exps = []
    for r in smart_rules:
        feat = extract_feature_name(r)
        if feat in mapping: exps.append(mapping[feat])
    return list(set(exps))

# =====================================================
# API ENDPOINT
# =====================================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    code = (await file.read()).decode("utf-8")
    metrics = extract_metrics(code)
    X = prepare_features(metrics)
    prob = float(model.predict_proba(X)[0][1])

    # 1. LIME Processing
    lime_raw_data = lime_explainer.explain_instance(X[0], model.predict_proba, num_features=10).as_list()
    full_lime_clean = [r for r, w in lime_raw_data if not (("<" in r and ">" in r) or (r.count('<') + r.count('>') > 1))]
    smart_lime = smart_select_lime(lime_raw_data, metrics)
    
    # 2. FORCED ANCHOR: Lowering threshold and increasing beam search to find results
    anchor_exp = anchor_explainer.explain(
        X[0], 
        threshold=0.60,      # Reduced from 0.95 to force a result
        beam_size=5,        # Increased to search more paths
        max_anchor_size=5    # Allow complex rules
    )
    anchor_rules = anchor_exp.anchor if anchor_exp.anchor else ["No specific anchor found"]
    
    # 3. COMBINATION: Add Smart LIME directly to the Anchor result list
    final_combined_rules = list(set(anchor_rules + smart_lime))

    return {
        "defect_probability": round(prob, 3),
        "severity": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "full_lime": full_lime_clean,
        "anchor_only": anchor_rules,
        "combined_anchor_result": final_combined_rules,
        "human_explanation": generate_human_explanation(smart_lime)
    }