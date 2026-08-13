import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2 - Classification App", page_icon="🤖", layout="wide")
st.title("Machine Learning Assignment 2: Classification Model Evaluation")
st.caption("Dataset: UCI Breast Cancer Wisconsin Diagnostic, 569 records, 30 numeric features, binary classification")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.sidebar.header("Upload and Select")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_name = st.sidebar.selectbox("Select ML model", list(MODEL_FILES.keys()))

@st.cache_data
def load_default_data():
    return pd.read_csv("test_data.csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using included test_data.csv for demonstration.")
    data = load_default_data()

st.subheader("Test Data Preview")
st.dataframe(data.head(15), use_container_width=True)

if "target" not in data.columns:
    st.error("The test CSV must contain a 'target' column for evaluation.")
    st.stop()

X = data.drop(columns=["target"])
y_true = data["target"]
model = joblib.load(MODEL_FILES[model_name])
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_score = model.predict_proba(X)[:, 1]
else:
    y_score = model.decision_function(X)

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "MCC Score": matthews_corrcoef(y_true, y_pred),
}

st.subheader(f"Evaluation Metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Malignant", "Benign"]); ax.set_yticklabels(["Malignant", "Benign"])
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_true, y_pred, target_names=["malignant", "benign"]))

st.subheader("Prediction Output")
output = data.copy()
output["predicted_target"] = y_pred
output["predicted_label"] = output["predicted_target"].map({0: "malignant", 1: "benign"})
st.dataframe(output.head(25), use_container_width=True)
