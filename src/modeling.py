from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data_generator import FEATURES

def train_and_save(df: pd.DataFrame, model_dir: str = "models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES], df["is_fraud"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    candidates = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=260, max_depth=13, min_samples_leaf=3,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ])
    }
    results, fitted = [], {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        })
        fitted[name] = model
    best_name = max(results, key=lambda r: r["roc_auc"])["model"]
    joblib.dump(fitted[best_name], Path(model_dir) / "fraud_classifier.joblib")
    joblib.dump(fitted, Path(model_dir) / "model_comparison.joblib")

    # Unsupervised outlier detector is trained on the feature space.
    anomaly = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(
            n_estimators=220, contamination=.035, random_state=42, n_jobs=-1
        ))
    ])
    anomaly.fit(X_train)
    joblib.dump(anomaly, Path(model_dir) / "anomaly_detector.joblib")
    pd.DataFrame(results).to_csv(Path(model_dir) / "model_metrics.csv", index=False)
    return best_name, pd.DataFrame(results), X_test, y_test

def score_transactions(df: pd.DataFrame, model, anomaly_model, threshold=.5):
    X = df[FEATURES]
    out = df.copy()
    out["fraud_probability"] = model.predict_proba(X)[:, 1]
    out["risk_score"] = (out["fraud_probability"] * 100).round(1)
    out["anomaly_flag"] = anomaly_model.predict(X) == -1
    out["decision"] = np.where(
        out["fraud_probability"] >= threshold, "Review / Block",
        np.where(out["anomaly_flag"], "Investigate", "Approve")
    )
    return out
