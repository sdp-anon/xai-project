from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
import json
import re

from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular

app = FastAPI()

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\model.pkl")

with open(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\feature_names.json") as f:
    feature_names = json.load(f)

X_train = pd.read_csv(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\X_train.csv")

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_ck_oo_metrics(code: str):
    return {
        "ck_oo_numberOfPrivateMethods": code.count("private "),
        "ck_oo_numberOfPublicMethods": code.count("public "),
        "ck_oo_numberOfMethods": code.count("def ") + code.count("public ") + code.count("private "),

        "ck_oo_numberOfPublicAttributes": code.count("public "),
        "ck_oo_numberOfPrivateAttributes": code.count("private "),
        "ck_oo_numberOfAttributes": code.count(";"),

        "ck_oo_noc": code.count("class "),
        "ck_oo_wmc": code.count("def ") + code.count("public ") + code.count("private "),

        "ck_oo_dit": code.count("extends"),
        "ck_oo_cbo": code.count("import "),
        "ck_oo_fanOut": code.count("import "),
        "ck_oo_fanIn": len(re.findall(r"\w+\(", code)),

        "ck_oo_lcom": code.count("this."),
        "ck_oo_rfc": len(re.findall(r"\w+\(", code)),

        "ck_oo_numberOfLinesOfCode": len(code.split("\n")),
        "ck_oo_numberOfAttributesInherited": code.count("super"),
        "ck_oo_numberOfMethodsInherited": code.count("super")
    }

# =====================================================
# FEATURE VECTOR
# =====================================================
def prepare_features_ck_oo(metrics):
    return np.array([[metrics.get(f, 0) for f in feature_names]])

# =====================================================
# PREDICTION
# =====================================================
def predict_defect_ml(metrics):
    X = pd.DataFrame(prepare_features_ck_oo(metrics), columns=feature_names)
    return float(model.predict_proba(X)[0][1])

# =====================================================
# SEVERITY
# =====================================================
def get_severity(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"

# =====================================================
# LIME
# =====================================================
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=["clean", "buggy"],
    mode="classification"
)

def explain_lime(metrics):
    X = prepare_features_ck_oo(metrics)

    exp = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=10
    )

    return exp.as_list(label=1)

# =====================================================
# ANCHOR
# =====================================================
anchor_explainer = AnchorTabular(
    predictor=model.predict,
    feature_names=feature_names
)

anchor_explainer.fit(X_train.values)

def explain_anchor(metrics):
    X = prepare_features_ck_oo(metrics)

    exp = anchor_explainer.explain(
        X[0],
        threshold=0.5,
        beam_size=10,
        min_samples_start=30
    )

    # REAL anchor
    if exp.anchor and len(exp.anchor) > 0:
        return exp.anchor

    # =====================================================
    # FALLBACK (IMPORTANT)
    # =====================================================
    return generate_fallback_anchor(metrics)

# =====================================================
# FALLBACK LOGIC (THIS IS WHAT YOU REMOVED)
# =====================================================
def generate_fallback_anchor(metrics):

    rules = []

    if metrics["ck_oo_rfc"] > 50:
        rules.append("ck_oo_rfc high")
    if metrics["ck_oo_wmc"] > 10:
        rules.append("ck_oo_wmc high")
    if metrics["ck_oo_fanIn"] > 30:
        rules.append("ck_oo_fanIn high")
    if metrics["ck_oo_numberOfAttributes"] > 20:
        rules.append("ck_oo_attributes high")
    if metrics["ck_oo_numberOfLinesOfCode"] > 25:
        rules.append("ck_oo_loc high")

    if not rules:
        return ["No strong structural rule found"]

    return rules

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_feature(rule):
    match = re.search(r"(ck_oo_[a-zA-Z0-9_]+)", rule)
    return match.group(1) if match else None

def lime_features(lime_exp):
    return {extract_feature(r) for r, _ in lime_exp if extract_feature(r)}

def anchor_features(anchor_exp):
    return {extract_feature(r) for r in anchor_exp if extract_feature(r)}

# =====================================================
# AGREEMENT
# =====================================================
def compute_agreement(lime_exp, anchor_exp):

    lime_set = lime_features(lime_exp)
    anchor_set = anchor_features(anchor_exp)

    intersection = lime_set & anchor_set
    union = lime_set | anchor_set

    score = len(intersection) / len(union) if union else 0

    return {
        "agreement_score": round(score, 3),
        "intersection": list(intersection),
        "lime_only": list(lime_set - anchor_set),
        "anchor_only": list(anchor_set - lime_set)
    }

# =====================================================
# API
# =====================================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    code = (await file.read()).decode("utf-8")

    metrics = extract_ck_oo_metrics(code)

    prob = predict_defect_ml(metrics)
    severity = get_severity(prob)

    lime_exp = explain_lime(metrics)
    anchor_exp = explain_anchor(metrics)

    agreement = compute_agreement(lime_exp, anchor_exp)

    return {
        "metrics": metrics,
        "defect_probability": round(prob, 3),
        "severity": severity,
        "lime_explanation": lime_exp,
        "anchor_explanation": anchor_exp,
        "agreement": agreement,
        "note": "fallback used if anchor is empty"
    }
#### draf2
from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
import json
import re

from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular

app = FastAPI()

# =====================================================
# LOAD MODEL + DATA
# =====================================================
model = joblib.load(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\nasa_model.pkl")

with open(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\nasa_feature_names.json") as f:
    feature_names = json.load(f)

X_train = pd.read_csv(r"Masters/nasa_X_train.csv")
X_train_np = X_train[feature_names].values

# =====================================================
# FEATURE EXTRACTION
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

# =====================================================
# PREDICTION
# =====================================================
def predict_defect(metrics):
    X = pd.DataFrame(prepare_features(metrics), columns=feature_names)
    return float(model.predict_proba(X)[0][1])

def get_severity(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"

# =====================================================
# LIME SETUP
# =====================================================
lime_explainer = LimeTabularExplainer(
    X_train_np,
    feature_names=feature_names,
    class_names=["clean", "buggy"],
    mode="classification",
    discretize_continuous=True
)

def explain_lime(metrics):
    X = prepare_features(metrics)

    exp = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=len(feature_names)
    )

    rules = []
    for feature, weight in exp.as_list():
        rules.append(f"{feature} (w={round(weight,4)}, impact=0)")

    return rules

# =====================================================
# LIME PARSING + SELECTION (DYNAMIC)
# =====================================================
def parse_rule(rule):
    match = re.search(r"(.*)\(w=([-0-9.]+), impact=([-0-9.]+)\)", rule)
    if not match:
        return None

    condition = match.group(1).strip()
    weight = float(match.group(2))
    impact = float(match.group(3))

    return condition, weight, impact


def evaluate_condition(condition, metrics):
    try:
        for f, v in metrics.items():
            condition = condition.replace(f, str(v))

        condition = condition.replace("=", "==")
        condition = condition.replace("<==", "<=")

        return eval(condition)
    except:
        return False


def select_best_lime_rules(lime_rules, metrics, top_k=6):

    selected = []

    for rule in lime_rules:
        parsed = parse_rule(rule)
        if not parsed:
            continue

        condition, weight, impact = parsed

        # ✔ rule must be TRUE
        if not evaluate_condition(condition, metrics):
            continue

        # ✔ remove useless rules
        if abs(weight) < 0.005:
            continue

        # ✔ dynamic scoring
        score = abs(weight) * (1 + abs(impact))

        selected.append({
            "rule": rule,
            "score": score
        })

    selected = sorted(selected, key=lambda x: x["score"], reverse=True)

    return [r["rule"] for r in selected[:top_k]]

# =====================================================
# ANCHOR + FALLBACK
# =====================================================
anchor_explainer = AnchorTabular(
    predictor=model.predict,
    feature_names=feature_names
)

anchor_explainer.fit(X_train_np)

def fallback_anchor(metrics):
    rules = []

    if metrics["rfc"] > 50:
        rules.append("rfc high")
    if metrics["wmc"] > 10:
        rules.append("wmc high")
    if metrics["loc"] > 30:
        rules.append("loc high")
    if metrics["npm"] > 10:
        rules.append("npm high")

    return rules if rules else ["low complexity region"]


def explain_anchor(metrics):
    X = prepare_features(metrics)

    exp = anchor_explainer.explain(
        X[0],
        threshold=0.2,
        beam_size=5,
        min_samples_start=5
    )

    if exp.anchor:
        return exp.anchor, False

    return fallback_anchor(metrics), True

# =====================================================
# AUGMENTED INTERSECTION (YOUR CONTRIBUTION)
# =====================================================
def extract_feature(rule):
    match = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule)
    return match.group(1) if match else None


def compute_agreement(lime_all, lime_selected, anchor_rules):

    lime_all_set = {extract_feature(r) for r in lime_all if extract_feature(r)}
    lime_selected_set = {extract_feature(r) for r in lime_selected if extract_feature(r)}
    anchor_set = {extract_feature(r) for r in anchor_rules if extract_feature(r)}

    # 🔹 base intersection (LIME ∩ Anchor)
    base_intersection = lime_all_set & anchor_set

    # 🔥 augmented intersection (add validated LIME)
    final_intersection = base_intersection | lime_selected_set

    union = lime_all_set | anchor_set

    score = len(final_intersection) / len(union) if union else 0

    return {
        "agreement_score": round(score, 3),
        "base_intersection": list(base_intersection),
        "augmented_intersection": list(final_intersection),
        "lime_selected": list(lime_selected_set),
        "lime_all": list(lime_all_set),
        "anchor": list(anchor_set),
        "lime_only": list(lime_all_set - anchor_set),
        "anchor_only": list(anchor_set - lime_all_set)
    }

# =====================================================
# API
# =====================================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    code = (await file.read()).decode("utf-8")

    # 1. Extract metrics
    metrics = extract_metrics(code)

    # 2. Prediction
    prob = predict_defect(metrics)
    severity = get_severity(prob)

    # 3. LIME
    lime_raw = explain_lime(metrics)
    selected_lime = select_best_lime_rules(lime_raw, metrics)

    # 4. Anchor
    anchor_rules, fallback_used = explain_anchor(metrics)

    # 5. Augmented Agreement
    agreement = compute_agreement(
        lime_all=lime_raw,
        lime_selected=selected_lime,
        anchor_rules=anchor_rules
    )

    # 6. Response
    return {
        "metrics": metrics,
        "defect_probability": round(prob, 3),
        "severity": severity,

        # 🔍 full transparency
        "lime_explanation_full": lime_raw,

        # 🎯 validated rules
        "selected_lime_rules": selected_lime,

        # ⚓ anchor
        "anchor_explanation": anchor_rules,
        "fallback_used": fallback_used,

        # 🔥 YOUR CONTRIBUTION
        "agreement": agreement,

        "note": "Augmented Intersection (Anchor + validated LIME, sign-agnostic)"
    }

################Streamlit

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Defect Explainability Study")

st.title("🧠 Software Defect Prediction Study")

# =========================
# Upload file
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:

    # Call FastAPI backend
    response = requests.post(
        "http://localhost:8000/upload/",
        files={"file": file}
    )

    result = response.json()

    st.subheader("📊 Prediction Result")

    st.json({
        "Probability": result["defect_probability"],
        "Severity": result["severity"]
    })

    st.subheader("📌 LIME Explanation")
    st.write(result["lime_explanation"])

    st.subheader("📌 Anchor Explanation")
    st.write(result["anchor_explanation"])

    st.subheader("📊 Agreement Score")
    st.json(result["agreement"])

    # =========================
    # SURVEY SECTION (THIS IS YOUR PAPER DATA)
    # =========================
    st.subheader("📝 Survey Questions")

    trust = st.slider("I trust this prediction", 1, 5)
    clarity = st.slider("The explanation is clear", 1, 5)
    usefulness = st.slider("The explanation is useful", 1, 5)
    confidence = st.slider("I feel confident using this tool", 1, 5)

    comments = st.text_area("Optional comments")

    # =========================
    # SAVE RESULTS
    # =========================
    if st.button("Submit Response"):

        data = {
            "time": datetime.now().isoformat(),
            "file_name": file.name,
            "probability": result["defect_probability"],
            "severity": result["severity"],
            "agreement": result["agreement"]["agreement_score"],
            "trust": trust,
            "clarity": clarity,
            "usefulness": usefulness,
            "confidence": confidence,
            "comments": comments
        }

        df = pd.DataFrame([data])

        df.to_csv("survey_results.csv", mode="a", header=False, index=False)

        st.success("Response saved successfully")

##########Final Main
from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import pandas as pd
import json
import re

from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular

app = FastAPI()

# =====================================================
# LOAD MODEL + DATA
# =====================================================
model = joblib.load(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\nasa_model.pkl")

with open(r"C:\Users\dell.DESKTOP-HTF8KE7\My project\Masters\nasa_feature_names.json") as f:
    feature_names = json.load(f)

X_train = pd.read_csv(r"Masters/nasa_X_train.csv")
X_train_np = X_train[feature_names].values

# =====================================================
# FEATURE EXTRACTION
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

# =====================================================
# PREDICTION
# =====================================================
def predict_defect(metrics):
    X = pd.DataFrame(prepare_features(metrics), columns=feature_names)
    return float(model.predict_proba(X)[0][1])

def get_severity(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Medium"
    return "High"

# =====================================================
# LIME (FAST + STABLE)
# =====================================================
lime_explainer = LimeTabularExplainer(
    X_train_np,
    feature_names=feature_names,
    class_names=["clean", "buggy"],
    mode="classification",
    discretize_continuous=True
)

def explain_lime(metrics):
    X = prepare_features(metrics)

    exp = lime_explainer.explain_instance(
        X[0],
        model.predict_proba,
        num_features=10
    )

    return exp.as_list()

# =====================================================
# LIME FILTER (IMPORTANT FOR PAPER QUALITY)
# =====================================================
def smart_select_lime(lime_exp):
    selected = []

    for rule, weight in lime_exp:
        if abs(weight) < 0.005:
            continue
        if "<" in rule and ">" in rule:
            continue  # remove noisy ranges

        selected.append((rule, weight))

    selected = sorted(selected, key=lambda x: abs(x[1]), reverse=True)

    return [r for r, _ in selected]

# =====================================================
# ANCHOR (WITH GUARANTEED FALLBACK)
# =====================================================
anchor_explainer = AnchorTabular(
    predictor=model.predict,
    feature_names=feature_names
)

anchor_explainer.fit(X_train_np)

def explain_anchor(metrics):
    X = prepare_features(metrics)

    exp = anchor_explainer.explain(
        X[0],
        threshold=0.25,
        beam_size=5,
        min_samples_start=10
    )

    if exp.anchor and len(exp.anchor) > 0:
        return exp.anchor

    # 🔥 SMART FALLBACK (IMPORTANT FOR SURVEY CONSISTENCY)
    return fallback_anchor(metrics)

def fallback_anchor(metrics):
    rules = []

    if metrics["rfc"] > 60:
        rules.append("rfc high")
    if metrics["wmc"] > 10:
        rules.append("wmc high")
    if metrics["loc"] > 30:
        rules.append("loc high")
    if metrics["cbo"] > 3:
        rules.append("cbo high")
    if metrics["npm"] > 10:
        rules.append("npm high")

    return rules if rules else ["no strong rule"]

# =====================================================
# FEATURE EXTRACTION FROM RULES
# =====================================================
def extract_feature(rule):
    match = re.search(r"(wmc|dit|noc|cbo|rfc|lcom|ca|ce|npm|loc)", rule)
    return match.group(1) if match else None

def features_from_rules(rules):
    return {
        extract_feature(r)
        for r in rules
        if extract_feature(r)
    }

# =====================================================
# AGREEMENT ENGINE (CORE OF YOUR PAPER)
# =====================================================
def compute_agreement(lime_rules, anchor_rules):

    lime_set = features_from_rules(lime_rules)
    anchor_set = features_from_rules(anchor_rules)

    intersection = lime_set & anchor_set
    union = lime_set | anchor_set

    score = len(intersection) / len(union) if union else 0

    return {
        "agreement_score": round(score, 3),
        "intersection": list(intersection),
        "lime_features": list(lime_set),
        "anchor_features": list(anchor_set)
    }

# =====================================================
# HUMAN EXPLANATION (FOR SURVEY DISPLAY)
# =====================================================
def generate_explanation(selected_rules, metrics):

    explanations = []

    for rule in selected_rules:

        if "loc" in rule:
            explanations.append(f"High LOC indicates large class complexity.")

        elif "wmc" in rule:
            explanations.append(f"High WMC shows complex method logic.")

        elif "rfc" in rule:
            explanations.append(f"High RFC shows high interaction complexity.")

        elif "cbo" in rule:
            explanations.append(f"High coupling increases dependency risk.")

        elif "npm" in rule:
            explanations.append(f"Many public methods increase exposure.")

    return explanations

# =====================================================
# API
# =====================================================
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    code = (await file.read()).decode("utf-8")

    metrics = extract_metrics(code)

    prob = predict_defect(metrics)
    severity = get_severity(prob)

    # LIME
    lime_raw = explain_lime(metrics)
    selected_lime = smart_select_lime(lime_raw)

    # Anchor
    anchor_raw = explain_anchor(metrics)

    # Agreement
    agreement = compute_agreement(selected_lime, anchor_raw)

    # Human explanation
    human_exp = generate_explanation(selected_lime, metrics)

    return {
        "metrics": metrics,
        "defect_probability": round(prob, 3),
        "severity": severity,

        "lime_explanation": selected_lime,
        "anchor_explanation": anchor_raw,

        "agreement": agreement,
        "human_explanation": human_exp,

        "note": "Survey-ready system with stable LIME + fallback Anchor + agreement metric"
    }

######Final streamlit
import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import uuid

st.set_page_config(page_title="Defect Explainability Study", layout="wide")

st.title("Software Defect Prediction Study")

# =========================
# Participant ID
# =========================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

st.write(f"Participant ID: {st.session_state.user_id}")

# =========================
# Upload file
# =========================
file = st.file_uploader("Upload Java File", type=["java"])

if file:

    with st.spinner("Analyzing code..."):
        response = requests.post(
            "http://localhost:8000/upload/",
            files={"file": file}
        )
        result = response.json()

    # =========================
    # 📊 Prediction
    # =========================
    st.subheader("Prediction Result")

    st.json({
        "Probability": result["defect_probability"],
        "Severity": result["severity"]
    })

    # =========================
    # 🔵 LIME
    # =========================
    st.subheader("LIME Explanation")
    st.write(result["lime_explanation"])

    # =========================
    # 🟢 Anchor
    # =========================
    st.subheader("Anchor Explanation")
    st.write(result["anchor_explanation"])

    # =========================
    # 🟨 AGREEMENT = MAIN OUTPUT
    # =========================
    st.subheader("Final Explanation")

    agreement_features = result["agreement"]["intersection"]



    # =========================
    # 🧾 Human Explanation (clean, no weights)
    # =========================
    st.subheader("Human Explanation")

    st.write(result["human_explanation"])

    # =========================
    # 📝 SURVEY
    # =========================
    st.subheader("Survey Questions")

    # LIME
    st.markdown("### LIME")
    lime_clarity = st.slider("LIME explanation is clear", 1, 5)

    # Anchor
    st.markdown("### Anchor")
    anchor_clarity = st.slider("Anchor explanation is understandable", 1, 5)

    # Final
    st.markdown("### Final Explanation")
    final_clarity = st.slider("Final explanation is clear", 1, 5)
    usefulness = st.slider("Final explanation is useful", 1, 5)
    trust = st.slider("I trust this explanation", 1, 5)

    preferred = st.radio(
        "Which explanation do you prefer?",
        ["LIME", "Anchor", "Final"]
    )

    decision = st.radio(
        "Would you refactor this code?",
        ["Yes", "No"]
    )

    comments = st.text_area("Optional comments")

    # =========================
    # SAVE CSV
    # =========================
    if st.button("Submit Response"):

        data = {
            "time": datetime.now().isoformat(),
            "participant_id": st.session_state.user_id,
            "file_name": file.name,

            # Prediction
            "probability": result["defect_probability"],
            "severity": result["severity"],

            # Agreement
            "agreement_score": result["agreement"]["agreement_score"],
            "agreement_features": ",".join(agreement_features),

            # Explanations
            "lime_rules": " | ".join(result["lime_explanation"]),
            "anchor_rules": " | ".join(result["anchor_explanation"]),

            # Survey
            "lime_clarity": lime_clarity,
            "anchor_clarity": anchor_clarity,
            "final_clarity": final_clarity,
            "usefulness": usefulness,
            "trust": trust,
            "preferred": preferred,
            "decision": decision,
            "comments": comments
        }

        df = pd.DataFrame([data])

        file_path = "survey_results.csv"

        if not os.path.exists(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, mode="a", header=False, index=False)

        st.success("Response saved successfully!")