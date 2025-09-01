"""
03_descriptives.py
- analysis_ready_expanded.csv.gz 로부터 기술통계, 그룹비교, 박스플롯 생성
- 산출물:
  - out/desc_continuous.csv
  - out/desc_categorical.csv
  - out/group_compare.csv
  - out/boxplots_group_comparison.png
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy import stats as st
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

pd.set_option("display.width", 160)


def welch_ttest_p(x0, x1):
    """Welch's t-test p-value (two-sided). SciPy 없으면 NaN 반환."""
    if not HAS_SCIPY:
        return np.nan
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    t, p = st.ttest_ind(x0, x1, equal_var=False)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="csv/analysis_ready_expanded.csv.gz")
    parser.add_argument("--outdir", type=str, default="out")
    args = parser.parse_args()

    Path(args.outdir).mkdir(exist_ok=True, parents=True)

    # -----------------------------
    # Load
    # -----------------------------
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded: {df.shape}")
    print("Columns:", list(df.columns))

    # 유도 변수 보완
    if "age10" not in df and "age" in df:
        df["age10"] = df["age"] / 10.0

    # -----------------------------
    # 변수 세트 (존재하는 것만 사용)
    # -----------------------------
    cont_candidates = ["age", "mvpa_min_day", "sed_ratio", "worn_min_day"]
    cont_vars = [c for c in cont_candidates if c in df.columns]

    cat_candidates = ["sex_female", "smoker", "alcohol_freq", "incm", "edu"]
    cat_vars = [c for c in cat_candidates if c in df.columns]

    outcomes_all = ["out_obesity", "out_hypertension", "out_diabetes"]
    outcomes = [o for o in outcomes_all if o in df.columns]

    # -----------------------------
    # 연속형 기술통계
    # -----------------------------
    if cont_vars:
        desc_cont = df[cont_vars].describe().T
        out_cont = os.path.join(args.outdir, "desc_continuous.csv")
        desc_cont.to_csv(out_cont)
        print("\n[Continuous variables]")
        print(desc_cont)
        print(f"[Saved] {out_cont}")
    else:
        print("\n[WARN] 연속형 변수 없음 → 스킵")

    # -----------------------------
    # 범주형 분포
    # -----------------------------
    if cat_vars:
        # 각 변수별 value_counts를 하나의 테이블로 정리
        cat_tables = []
        for c in cat_vars:
            vc = df[c].value_counts(dropna=False).rename("count").to_frame()
            vc["variable"] = c
            vc["level"] = vc.index
            cat_tables.append(vc.reset_index(drop=True))
        desc_cat = pd.concat(cat_tables, ignore_index=True)
        # 보기 편하게 pivot(옵션)
        pivot_cat = desc_cat.pivot_table(index=["variable", "level"], values="count", aggfunc="sum")
        out_cat = os.path.join(args.outdir, "desc_categorical.csv")
        pivot_cat.to_csv(out_cat)
        print("\n[Categorical variables]")
        for c in cat_vars:
            print(f"\n{c}:")
            print(df[c].value_counts(dropna=False))
        print(f"[Saved] {out_cat}")
    else:
        print("\n[WARN] 범주형 변수 없음 → 스킵")

    # -----------------------------
    # Outcome별 그룹 비교 (연속형만 평균 비교)
    # -----------------------------
    rows = []
    for oc in outcomes:
        if df[oc].notna().sum() == 0:
            continue
        for var in cont_vars:
            # 그룹 분할
            g0 = df.loc[df[oc] == 0, var]
            g1 = df.loc[df[oc] == 1, var]
            mean0 = np.nanmean(g0) if g0.notna().any() else np.nan
            mean1 = np.nanmean(g1) if g1.notna().any() else np.nan
            pval = welch_ttest_p(g0, g1)
            rows.append({
                "outcome": oc,
                "variable": var,
                "mean_no": mean0,
                "mean_yes": mean1,
                "p_value": pval
            })
    if rows:
        cmp_df = pd.DataFrame(rows)
        out_cmp = os.path.join(args.outdir, "group_compare.csv")
        cmp_df.to_csv(out_cmp, index=False)
        print(f"\n[Saved] {out_cmp}")
        # 콘솔 요약
        with pd.option_context("display.float_format", "{:,.6f}".format):
            print(cmp_df)
    else:
        print("\n[WARN] 그룹 비교 결과 없음 (연속형/아웃컴 확인 필요)")

    # -----------------------------
    # 박스플롯 (경고 해결: hue를 x에 지정, legend=False)
    #  - 각 outcome마다 핵심 연속형 변수들의 박스플롯
    # -----------------------------
    if outcomes and cont_vars:
        plot_pairs = []
        # 대표 변수 소수만 선택(그림 크기 과도 확대 방지)
        pick = [v for v in ["mvpa_min_day", "sed_ratio", "age", "worn_min_day"] if v in cont_vars]
        if not pick:
            pick = cont_vars[:3]

        # subplot 갯수
        n_oc = len(outcomes)
        n_var = len(pick)
        n_sub = n_oc * n_var
        ncol = min(3, n_var)
        nrow = int(np.ceil(n_sub / ncol))

        fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.0 * nrow))
        if nrow * ncol == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        idx = 0
        for oc in outcomes:
            # outcome 이진만 박스플롯 의미 있음
            if df[oc].dropna().nunique() < 2:
                continue
            for var in pick:
                ax = axes[idx]
                sns.boxplot(
                    data=df[[oc, var]].dropna(),
                    x=oc,
                    y=var,
                    hue=oc,          # ★ 경고 해결
                    legend=False,    # 범례 숨김
                )
                ax.set_title(f"{var} by {oc}")
                ax.set_xlabel(oc)
                ax.set_ylabel(var)
                idx += 1
                if idx >= len(axes):
                    break

        # 빈 축 숨기기
        for j in range(idx, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out_fig = os.path.join(args.outdir, "boxplots_group_comparison.png")
        plt.savefig(out_fig, dpi=300)
        plt.close()
        print(f"[Saved] {out_fig}")
    else:
        print("\n[WARN] 박스플롯 스킵 (outcome 또는 연속형 변수 부족)")

    print("=== Done ===")


if __name__ == "__main__":
    main()
