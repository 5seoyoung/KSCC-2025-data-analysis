"""
04_models.py  (SES-안정화 핫픽스 v2)
- C(incm), C(edu)를 1~4 유효값으로 정리하고 '카테고리형'으로 강제
- 연속/더미 변수 및 결과(outcomes), 가중치(wt_tot)는 모두 float으로 강제
- C(incm), C(edu)에서 유효범주가 2개 미만이면 자동 제외 → 실패 시 SES 전체 제외 재시도
- OR 요약(out/summary_or_exposures.csv), 포리스트(out/forest_weighted_all.png)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import matplotlib.pyplot as plt

INP = "csv/analysis_ready_expanded.csv.gz"
OUT_CSV = "out/summary_or_exposures.csv"
OUT_FIG = "out/forest_weighted_all.png"

OUTCOMES = ["out_obesity", "out_hypertension", "out_diabetes"]
BASE_TERMS = ["mvpa10", "sed10", "age10", "sex_female"]   # 필수 공변량
SES_TERMS  = ["C(incm)", "C(edu)"]                       # 선택: 자동 점검

def ensure_dirs():
    os.makedirs("out", exist_ok=True)

def sanitize_ses(df: pd.DataFrame) -> pd.DataFrame:
    """incm/edu를 1~4만 유효로, 카테고리형으로 강제."""
    for c in ["incm", "edu"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            # 1~4만 유지, 나머지는 NaN
            s = s.where(s.isin([1, 2, 3, 4]))
            # 카테고리로 캐스팅 (ordered=True도 가능)
            df[c] = pd.Categorical(s, categories=[1, 2, 3, 4])
    return df

def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """파생/스케일 변환(있으면 유지) + 숫자형 강제."""
    if "age" in df and "age10" not in df.columns:
        df["age10"] = pd.to_numeric(df["age"], errors="coerce") / 10.0
    if "mvpa10" not in df.columns and "mvpa_min_day" in df.columns:
        df["mvpa10"] = pd.to_numeric(df["mvpa_min_day"], errors="coerce") / 10.0
    if "sed10" not in df.columns and "sed_ratio" in df.columns:
        df["sed10"] = pd.to_numeric(df["sed_ratio"], errors="coerce") * 10.0

    # 안전: 연속/더미형은 float로 강제
    for c in ["mvpa10", "sed10", "age10", "sex_female", "wt_tot"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    # 결과(outcomes)도 float로 강제
    for c in ["out_obesity", "out_hypertension", "out_diabetes"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df

def usable_categoricals(df: pd.DataFrame, terms: list) -> list:
    """C(x) 항 중 실제로 범주가 2개 이상인 항만 남김."""
    keep = []
    for t in terms:
        if not (t.startswith("C(") and t.endswith(")")):
            continue
        col = t[2:-1]
        if col in df.columns:
            vals = pd.Series(df[col]).dropna().unique()
            if len(vals) >= 2:
                keep.append(t)
    return keep

def fit_weighted_glm(df, outcome, rhs_terms, weight_col="wt_tot"):
    """가중 로지스틱 GLM. 가중치 결측은 1.0으로 대체."""
    w = df[weight_col] if weight_col in df.columns else pd.Series(1.0, index=df.index)
    w = pd.to_numeric(w, errors="coerce").fillna(1.0)
    formula = f"{outcome} ~ {' + '.join(rhs_terms)}"
    y, X = dmatrices(formula, data=df, return_type="dataframe")
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w.loc[y.index])
    res = model.fit()
    return res

def extract_or(res, label_map: dict):
    """회귀 결과에서 OR/CI/P 값 추출."""
    coefs = res.params
    covs  = res.cov_params()
    out = []
    for name in coefs.index:
        if name == "Intercept":
            continue
        beta = coefs[name]
        se   = np.sqrt(covs.loc[name, name]) if name in covs.index else np.nan
        OR   = np.exp(beta)
        lo   = np.exp(beta - 1.96*se) if pd.notna(se) else np.nan
        hi   = np.exp(beta + 1.96*se) if pd.notna(se) else np.nan
        pval = res.pvalues.get(name, np.nan)
        lab  = label_map.get(name, name)
        out.append({"term": name, "label": lab, "OR": OR, "CI_low": lo, "CI_high": hi, "p": pval})
    return pd.DataFrame(out)

def forest_plot(df_or: pd.DataFrame, path_png: str, title: str = "Weighted ORs"):
    d = df_or.copy().dropna(subset=["OR"])
    d = d.sort_values("OR")
    fig, ax = plt.subplots(figsize=(6, 0.4*len(d)+1.5))
    ax.errorbar(
        d["OR"], range(len(d)),
        xerr=[d["OR"]-d["CI_low"], d["CI_high"]-d["OR"]],
        fmt="o", capsize=3
    )
    ax.axvline(1.0, ls="--", lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_xscale("log")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)

def main():
    ensure_dirs()
    df = pd.read_csv(INP)
    print("Loading:", INP)
    print("Columns:", list(df.columns))

    # 변환/형 강제 → SES 카테고리화
    df = add_transforms(df)
    df = sanitize_ses(df)

    # 용어 라벨
    label_map = {
        "mvpa10": "MVPA (per 10 min)",
        "sed10" : "Sedentary (per 10%-pt)",
        "age10" : "Age (per 10y)",
        "sex_female": "Female (vs male)",
    }

    all_rows = []
    for y in OUTCOMES:
        # 필수 변수 결측 제거(결과/노출/핵심공변량)
        base_needed = [y, "mvpa10", "sed10", "age10", "sex_female"]
        d0 = df.dropna(subset=[c for c in base_needed if c in df.columns]).copy()

        # SES 항 중 유효한 것만 남김
        ses_ok = usable_categoricals(d0, SES_TERMS)
        rhs = BASE_TERMS.copy()
        rhs += ses_ok

        print(f"\n[Preflight] outcome={y}")
        cols_to_show = [c for c in ["out_obesity","out_hypertension","out_diabetes",
                                    "mvpa10","sed10","age10","sex_female"] if c in d0.columns]
        print(d0[cols_to_show].describe())

        tried = []
        res = None

        # 1차: SES 포함
        if len(ses_ok) > 0:
            try:
                res = fit_weighted_glm(d0, y, rhs, weight_col="wt_tot")
                tried.append(("with_ses", rhs))
            except Exception as e:
                print(f"[WARN] {y} 모델 실패(SES 포함): {e}")

        # 2차: SES 제거
        if res is None:
            try:
                rhs2 = BASE_TERMS.copy()
                res = fit_weighted_glm(d0, y, rhs2, weight_col="wt_tot")
                tried.append(("no_ses", rhs2))
            except Exception as e:
                print(f"[ERROR] {y} 모델 실패(SES 제외): {e}")
                continue

        # 결과 정리
        or_tab = extract_or(res, label_map)
        or_tab.insert(0, "outcome", y)
        all_rows.append(or_tab)

    if not all_rows:
        print("[ERROR] 유효한 모델 결과가 없습니다.")
        return

    summary = pd.concat(all_rows, ignore_index=True)
    summary.to_csv(OUT_CSV, index=False)
    print("Saved →", OUT_CSV)

    # Forest (노출 2개만 필터)
    forest_rows = summary[summary["term"].isin(["mvpa10","sed10"])].copy()
    if not forest_rows.empty:
        # outcome 포함 라벨로 포리스트
        forest_rows["label"] = forest_rows["label"] + " | " + forest_rows["outcome"].str.replace("out_","")
        forest_plot(forest_rows, OUT_FIG, title="Exposure ORs by outcome")
        print("Saved →", OUT_FIG)
    else:
        print("[WARN] forest 대상 행이 없습니다.")

if __name__ == "__main__":
    main()
