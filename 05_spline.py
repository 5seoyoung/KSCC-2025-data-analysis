"""
05_spline.py (fixed)
- analysis_ready_expanded.csv.gz 기반
- MVPA(min/day), Sedentary ratio 에 대해 RCS(Restricted Cubic Splines) 가중 로지스틱 회귀
- 아웃컴: out_obesity, out_hypertension, out_diabetes
- 공변량: age10, sex_female, (선택) C(incm), C(edu)
- 산출: 노출별 그림 PNG + 예측값 CSV
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices  # ★ 핵심: design_info 재활용

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def log(*args):
    print(*args, flush=True)

def ensure_dir(path):
    Path(path).mkdir(exist_ok=True, parents=True)

def clip_iqr(x, lower_q=0.02, upper_q=0.98):
    lo, hi = np.nanpercentile(x, [lower_q*100, upper_q*100])
    return lo, hi

def as_float(s):
    return pd.to_numeric(s, errors="coerce").astype(float)

# --------------------------------------------------------------------------------------
# Modeling helpers
# --------------------------------------------------------------------------------------

def build_design(df, exposure, df_spline=4, covars=None, use_ses=True):
    """
    RCS를 patsy dmatrix로 구성. SES 사용 시 incm/edu는 C()로만 처리(판다스 캐스팅 X).
    """
    covars = covars or []
    base_terms = []
    for c in covars:
        if c in ("incm", "edu"):
            if use_ses and c in df.columns:
                base_terms.append(f"C({c})")   # ← 이것만이면 충분
        else:
            if c in df.columns:
                base_terms.append(c)

    spline = f"cr({exposure}, df={df_spline})"
    rhs = " + ".join([spline] + base_terms) if base_terms else spline
    return rhs

def fit_glm_rcs(df, outcome, exposure, df_spline=4, covars=None, use_ses=True, weight_col="wt_tot"):
    """
    GLM Binomial + freq_weights
    """
    data = df.copy()

    # 결과/노출 수치화
    data[outcome] = as_float(data[outcome])
    data[exposure] = as_float(data[exposure])

    # 공변량 타입 정리
    if "age10" not in data.columns and "age" in data.columns:
        data["age10"] = as_float(data["age"]) / 10.0
    if "age10" in data.columns:
        data["age10"] = as_float(data["age10"])
    if "sex_female" in data.columns:
        data["sex_female"] = as_float(data["sex_female"])

    # 가중치
    if weight_col not in data.columns:
        data[weight_col] = 1.0
    data[weight_col] = as_float(data[weight_col]).fillna(1.0)

    # 설계 행렬
    rhs = build_design(data, exposure, df_spline=df_spline, covars=covars, use_ses=use_ses)
    X = dmatrix(rhs, data, return_type="dataframe")

    # 결측 제거 동기화
    keep_mask = (~data[outcome].isna()) & X.notnull().all(axis=1) & (~data[weight_col].isna())
    y_fit = data.loc[keep_mask, outcome].values
    X_fit = X.loc[keep_mask]
    w_fit = data.loc[keep_mask, weight_col].values

    # 사건 양쪽 존재 확인
    if y_fit.sum() == 0 or (len(y_fit) - y_fit.sum()) == 0:
        raise RuntimeError(f"[{outcome}] 사건(0/1)이 한쪽만 존재 → 모델 불가")

    # GLM (Binomial)
    model = sm.GLM(y_fit, X_fit, family=sm.families.Binomial(), freq_weights=w_fit)
    res = model.fit()
    return res, X.design_info, keep_mask, rhs  # ★ rhs도 반환

def predict_curve(res, design_info, rhs, df_ref, exposure, grid, use_ses=True):
    """
    학습 design_info로 예측용 X를 만들고, statsmodels의 get_prediction으로
    예측값/신뢰구간을 안정적으로 계산.
    """
    new = pd.DataFrame({exposure: grid})

    # 공변량 고정(연속=평균, 이진=round(mean), 범주형=최빈값)
    if "age10" in df_ref.columns:
        new["age10"] = pd.to_numeric(df_ref["age10"], errors="coerce").mean()
    elif "age" in df_ref.columns:
        new["age10"] = (pd.to_numeric(df_ref["age"], errors="coerce") / 10.0).mean()

    if "sex_female" in df_ref.columns:
        new["sex_female"] = int(round(pd.to_numeric(df_ref["sex_female"], errors="coerce").mean()))

    if use_ses:
        for c in ("incm", "edu"):
            if c in df_ref.columns:
                s = pd.to_numeric(df_ref[c], errors="coerce").dropna()
                new[c] = (int(s.mode().iloc[0]) if len(s) else 1)

    # 학습 design_info로 예측 설계행렬 구성
    X_new = build_design_matrices([design_info], new)[0]

    # 예측 (응답척도에서의 평균과 95% CI)
    pr = res.get_prediction(X_new)
    sf = pr.summary_frame()  # columns: mean, mean_se, mean_ci_lower, mean_ci_upper

    out = pd.DataFrame({
        exposure: grid,
        "pred":  sf["mean"].values,
        "lo95":  sf["mean_ci_lower"].values,
        "hi95":  sf["mean_ci_upper"].values,
    })
    return out

# --------------------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------------------

def plot_exposure_curves(exposure, curves_dict, xlabel, outfile):
    ensure_dir("out")
    outcomes = list(curves_dict.keys())

    fig, axes = plt.subplots(1, len(outcomes), figsize=(6 * len(outcomes), 4), sharey=True)
    if len(outcomes) == 1:
        axes = [axes]

    for ax, y in zip(axes, outcomes):
        c = curves_dict[y]
        ax.plot(c[exposure], c["pred"], label=y.replace("out_", ""))
        ax.fill_between(c[exposure], c["lo95"], c["hi95"], alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Predicted probability")
        ax.set_title(y.replace("out_", "").title())
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Spline curves: {exposure}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    log(f"[Saved] {outfile}")

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="csv/analysis_ready_expanded.csv.gz")
    parser.add_argument("--exposures", nargs="+", default=["mvpa_min_day", "sed_ratio"])
    parser.add_argument("--outcomes", nargs="+", default=["out_obesity", "out_hypertension", "out_diabetes"])
    parser.add_argument("--df", type=int, default=4, help="RCS degrees of freedom")
    parser.add_argument("--no-ses", action="store_true", help="SES(incm, edu) 제외")
    args = parser.parse_args()

    use_ses = not args.no_ses

    log(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    log("Columns:", list(df.columns))

    # 타입/파생 정리
    if "age10" not in df.columns and "age" in df.columns:
        df["age10"] = as_float(df["age"]) / 10.0
    if "sex_female" in df.columns:
        df["sex_female"] = as_float(df["sex_female"])
    if "wt_tot" in df.columns:
        df["wt_tot"] = as_float(df["wt_tot"]).fillna(1.0)
    else:
        df["wt_tot"] = 1.0

    for exp in args.exposures:
        if exp not in df.columns:
            log(f"[WARN] 노출 변수 없음: {exp} → 스킵")
            continue

        x = as_float(df[exp])
        lo, hi = clip_iqr(x, 0.02, 0.98)
        grid = np.linspace(lo, hi, 150)

        curves = {}
        for y in args.outcomes:
            if y not in df.columns:
                log(f"[WARN] outcome 없음: {y} → 스킵")
                continue

            try:
                res, design_info, keep_mask, rhs = fit_glm_rcs(
                    df, outcome=y, exposure=exp,
                    df_spline=args.df,
                    covars=["age10", "sex_female", "incm", "edu"],
                    use_ses=use_ses,
                    weight_col="wt_tot",
                )
            except Exception as e:
                log(f"[WARN] {y} 모델 실패: {e}")
                continue

            curve = predict_curve(
                res, design_info, rhs, df.loc[keep_mask],
                exposure=exp, grid=grid, use_ses=use_ses
            )
            curves[y] = curve

            csv_out = f"out/spline_pred_{exp}_{y}.csv"
            curve.to_csv(csv_out, index=False)
            log(f"[Saved] {csv_out}")

        if curves:
            xlabel = "MVPA (min/day)" if exp == "mvpa_min_day" else ("Sedentary ratio" if exp == "sed_ratio" else exp)
            png_out = f"out/spline_{exp}.png"
            plot_exposure_curves(exp, curves, xlabel, png_out)

    log("=== Done ===")

if __name__ == "__main__":
    ensure_dir("out")
    main()
