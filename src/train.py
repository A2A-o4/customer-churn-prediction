"""Train and evaluate churn models from the command line.

Usage: python src/train.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df.drop(columns="customerID")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["n_services"] = (d[SERVICES] == "Yes").sum(axis=1)
    d["tenure_group"] = pd.cut(d["tenure"], [-1, 6, 12, 24, 48, 72],
                               labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype(str)
    d["avg_monthly_spend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"].clip(lower=1), d["MonthlyCharges"])
    d["is_new_customer"] = (d["tenure"] <= 6).astype(int)
    d["no_protection"] = ((d["OnlineSecurity"] != "Yes") & (d["TechSupport"] != "Yes")).astype(int)
    return d


def main() -> None:
    df = add_features(load_data(ROOT / "data" / "Telco-Customer-Churn.csv"))
    X, y = df.drop(columns="Churn"), df["Churn"]
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                                random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_w, random_state=SEED, eval_metric="logloss"),
    }
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_tr, y_tr)
        proba, pred = pipe.predict_proba(X_te)[:, 1], pipe.predict(X_te)
        print(f"{name:20s} AUC={roc_auc_score(y_te, proba):.3f}  F1={f1_score(y_te, pred):.3f}  "
              f"Recall={recall_score(y_te, pred):.3f}  Precision={precision_score(y_te, pred):.3f}")


if __name__ == "__main__":
    main()
