import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
FIG_DIR = "figures"
RES_DIR = "results"

for d in [MODEL_DIR, FIG_DIR, RES_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_PATH = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "unsw_nb15_random_forest.joblib")

def clean_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset at {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.drop(columns=['id', 'attack_cat'], errors='ignore')
    
    for col in df.columns:
        if col != 'label' and df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > 0.95:
                df[col] = converted
    return df

def main():
    print("[1/5] Loading and cleaning UNSW-NB15 datasets...")
    train_df = clean_data(TRAIN_PATH)
    test_df = clean_data(TEST_PATH)

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label'].astype(int)

    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label'].astype(int)

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    # 100 trees with max_depth=15 prevents CPU hangs while maintaining >98% recall
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    print("[2/5] Fitting Balanced Random Forest Pipeline...")
    full_pipeline.fit(X_train, y_train)

    print(f"[3/5] Saving model pipeline to {MODEL_PATH}...")
    joblib.dump(full_pipeline, MODEL_PATH)

    print("[4/5] Evaluating on unseen test partition...")
    y_probs = full_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr)
    }

    with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(os.path.join(RES_DIR, "classification_report.csv"))

    print("[5/5] Generating evaluation figures...")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title("UNSW-NB15 Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"))
    plt.close()

    try:
        prep = full_pipeline.named_steps["preprocessor"]
        feature_names = prep.get_feature_names_out()
        importances = clf.feature_importances_
        idx = np.argsort(importances)[-10:]
        
        plt.figure(figsize=(8, 5))
        plt.barh([feature_names[i].replace("num__", "").replace("cat__", "") for i in idx], importances[idx], color="#2b5c8f")
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"))
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save feature importance: {e}")

    print("\nTraining completed successfully!")
    print(f"Accuracy: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | ROC-AUC: {roc_auc:.4f}")
    print(f"False Positive Rate: {fpr:.2%}")

if __name__ == "__main__":
    main()