"""
06_ml_shap.py
- XGBoost 분류기 + SHAP 해석
- 아웃컴별로 SHAP bar / beeswarm / dependence plot 저장
- 대표본(out_obesity)을 shap_bar.png, shap_beeswarm.png, shap_dependence.png 로 복사
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from shutil import copyfile

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 140

OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

OUTCOMES = ["out_obesity", "out_hypertension", "out_diabetes"]
BASE_FEATURES = [
    "mvpa_min_day", "sed_ratio", "age", "sex_female",
    "incm", "edu", "smoking", "alcohol", "kcal"
]

def load_data(path, max_n=None):
    df = pd.read_csv(path)
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=42).reset_index(drop=True)
    return df

def prep_X_y(df, outcome, features):
    df = df.copy()
    y = pd.to_numeric(df[outcome], errors="coerce")
    X = df[features].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)

    cat_cols = [c for c in ["incm","edu","sex_female","smoking"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop"
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4
    )

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", model)
    ])

    return X, y, pipe, preproc, num_cols, cat_cols

def train_eval(X, y, pipe, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    return pipe, auc, (Xtr, Xte, ytr, yte)

def to_design_matrix(preproc, X):
    Xt = preproc.transform(X)
    num_cols = preproc.transformers_[0][2]
    cat_trans, cat_cols = preproc.transformers_[1][1], preproc.transformers_[1][2]
    cat_names = []
    if isinstance(cat_trans, OneHotEncoder):
        cat_names = cat_trans.get_feature_names_out(cat_cols).tolist()
    cols = list(num_cols) + list(cat_names)
    return Xt, cols

def top_feature_from_shap(shap_values):
    vals = np.abs(shap_values.values).mean(axis=0)
    idx = int(np.argmax(vals))
    return shap_values.feature_names[idx]

def shap_plots(model, X_mat, feature_names, outcome):
    """SHAP 그림 저장 (bar / beeswarm / dependence) - X_mat은 ndarray, feature_names는 리스트"""
    # ✅ 피처명을 유지하기 위해 DataFrame으로 변환
    X_df = pd.DataFrame(X_mat, columns=feature_names)

    # 트리 기반 모델이므로 TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)  # Explanation 객체 (feature_names 포함)

    # --- BAR ---
    plt.figure(figsize=(6, 4.2))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance — {outcome}")
    bar_path = os.path.join(OUT_DIR, f"shap_bar_{outcome}.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    # --- BEESWARM ---
    plt.figure(figsize=(6.8, 4.8))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {outcome}")
    bee_path = os.path.join(OUT_DIR, f"shap_beeswarm_{outcome}.png")
    plt.tight_layout()
    plt.savefig(bee_path)
    plt.close()

    # --- DEPENDENCE ---
    # 기본 후보: mvpa_min_day / sed_ratio (OHE가 아니어서 이름 그대로 존재)
    fnames = list(shap_values.feature_names)
    main_feat = "mvpa_min_day" if "mvpa_min_day" in fnames else top_feature_from_shap(shap_values)
    color_feat = "sed_ratio" if "sed_ratio" in fnames else None

    plt.figure(figsize=(6.2, 4.4))
    if color_feat:
        shap.plots.scatter(
            shap_values[:, main_feat],
            color=shap_values[:, color_feat],
            show=False
        )
        plt.title(f"Dependence: {main_feat} (color={color_feat}) — {outcome}")
    else:
        shap.plots.scatter(shap_values[:, main_feat], show=False)
        plt.title(f"Dependence: {main_feat} — {outcome}")
    dep_path = os.path.join(OUT_DIR, f"shap_dependence_{outcome}.png")
    plt.tight_layout()
    plt.savefig(dep_path)
    plt.close()

    # 글로벌 중요도 CSV
    imp = pd.DataFrame({
        "feature": shap_values.feature_names,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    imp_path = os.path.join(OUT_DIR, f"shap_importance_{outcome}.csv")
    imp.to_csv(imp_path, index=False)

    return bar_path, bee_path, dep_path, imp_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="csv/analysis_ready_expanded.csv.gz")
    ap.add_argument("--max-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_data(args.data, max_n=args.max_n)
    print(f"Loaded: {df.shape}")
    print("Columns:", list(df.columns))
    features = [c for c in BASE_FEATURES if c in df.columns]
    print("Features used:", features)

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"[SKIP] {outcome} not found.")
            continue

        X, y, pipe, preproc, num_cols, cat_cols = prep_X_y(df, outcome, features)
        if len(X) < 100 or y.sum() == 0 or y.sum() == len(y):
            print(f"[SKIP] {outcome} — 데이터 또는 양성/음성 불균형으로 스킵 (N={len(X)}, pos={y.sum()}).")
            continue

        pipe, auc, (Xtr, Xte, ytr, yte) = train_eval(X, y, pipe, seed=args.seed)
        print(f"\n=== Outcome: {outcome} ===")
        print(f"Data: ({len(X)}, {X.shape[1]}) | Pos rate: {y.mean():.4f}")
        print(f"AUC(test) = {auc:.3f}")

        preproc_fitted = pipe.named_steps["prep"]
        Xt_full, feat_names = to_design_matrix(preproc_fitted, X)
        model = pipe.named_steps["clf"]

        bar_path, bee_path, dep_path, imp_path = shap_plots(model, Xt_full, feat_names, outcome)

        # 대표본(out_obesity)을 공모전 파일명으로 복사
        if outcome == "out_obesity":
            copyfile(bar_path, os.path.join(OUT_DIR, "shap_bar.png"))
            copyfile(bee_path, os.path.join(OUT_DIR, "shap_beeswarm.png"))
            copyfile(dep_path, os.path.join(OUT_DIR, "shap_dependence.png"))

    print("=== Done (SHAP) ===")

if __name__ == "__main__":
    main()
