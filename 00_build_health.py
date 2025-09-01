"""
00_build_health.py
KNHANES ALL 데이터 병합 및 health_2014_2017.csv.gz 생성
- 연도별 csv/{year}_all.csv.gz 를 읽어 표준 스키마로 정리
- 임상/검사(BMI, SBP, DBP, HbA1c)로 아웃컴 산출
- 약물변수 있으면 포함, 없으면 검사치 기준만으로 산출
- SES/가중치가 있으면 함께 보존
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)

# 1) 표준 스키마 → 실제 파일에서 찾을 후보 컬럼명 목록
#    (필요하면 오른쪽 후보 리스트에 실제 이름을 추가하세요)
CANDIDATES = {
    "ID"      : ["ID", "id"],
    "year"    : ["year", "YEAR"],
    "sex"     : ["sex", "SEX"],
    "age"     : ["age", "AGE", "age_year"],

    # 임상/검사
    "BMI"     : ["HE_BMI", "bmi", "BMI"],
    "SBP"     : ["HE_sbp", "SBP", "sbp", "he_SBP"],
    "DBP"     : ["HE_dbp", "DBP", "dbp", "he_DBP"],
    "HbA1c"   : ["HE_HbA1c", "HE_HBA1C", "HbA1c", "HBA1C"],

    # 약물 (있을 때만 사용)
    "htn_med" : ["HE_HPDR", "htn_med", "BP_MED", "bp_med"],
    "dm_med"  : ["HE_DMDR", "dm_med", "DM_MED"],

    # 생활습관/영양 (있으면 보존)
    "smoking" : ["sm_presnt", "SM_PRESNT", "smoking"],
    "alcohol" : ["dr_month", "DR_MONTH", "alcohol"],
    "kcal"    : ["N_EN", "EN", "kcal"],

    # SES/가중치 (있으면 보존)
    "incm"    : ["incm", "INCM"],
    "edu"     : ["edu", "EDU"],
    "wt_tot"  : ["wt_tot", "WT_TOT", "wt"],  # 가중치 후보
    "psu"     : ["psu", "PSU"],
    "kstrata" : ["kstrata", "KSTRATA", "strata", "STRATA"],
}

# --------------------------------------------------------------------------------------

def find_first_present(df: pd.DataFrame, candidates) -> str | None:
    """데이터프레임에서 후보 컬럼들 중 첫 번째로 존재하는 컬럼명을 반환(대소문자 무시)."""
    if isinstance(candidates, str):
        candidates = [candidates]
    lowers = {c.lower(): c for c in df.columns}
    for cand in candidates or []:
        if cand in df.columns:
            return cand
        hit = lowers.get(str(cand).lower())
        if hit is not None:
            return hit
    return None

def read_all_standardized(path_csv_gz: str) -> pd.DataFrame:
    """연도별 ALL 파일을 읽어 표준 스키마로 정리."""
    df = pd.read_csv(path_csv_gz, low_memory=False)
    out = pd.DataFrame()

    # ID / year (필수)
    id_col   = find_first_present(df, CANDIDATES["ID"])
    year_col = find_first_present(df, CANDIDATES["year"])
    if id_col is None or year_col is None:
        raise ValueError(f"[{path_csv_gz}] ID/year 컬럼을 찾을 수 없습니다.")
    out["ID"]   = df[id_col].astype(str)
    out["year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    # 기본 인구학
    for k in ["sex", "age"]:
        col = find_first_present(df, CANDIDATES[k])
        if col:
            out[k] = pd.to_numeric(df[col], errors="coerce")

    # 임상/검사
    for k in ["BMI", "SBP", "DBP", "HbA1c"]:
        col = find_first_present(df, CANDIDATES[k])
        if col:
            out[k] = pd.to_numeric(df[col], errors="coerce")

    # 약물(있을 때만)
    for k in ["htn_med", "dm_med"]:
        col = find_first_present(df, CANDIDATES[k])
        if col:
            out[k] = (pd.to_numeric(df[col], errors="coerce") == 1).astype("Int64")

    # 생활습관/영양(있으면 보존)
    for k in ["smoking", "alcohol", "kcal"]:
        col = find_first_present(df, CANDIDATES[k])
        if col:
            out[k] = pd.to_numeric(df[col], errors="coerce")

    # SES/가중치(있으면 보존)
    for k in ["incm", "edu", "wt_tot", "psu", "kstrata"]:
        col = find_first_present(df, CANDIDATES[k])
        if col:
            out[k] = pd.to_numeric(df[col], errors="coerce")

    return out

def build_health(years, obesity_cut=25.0) -> pd.DataFrame:
    Path("csv").mkdir(exist_ok=True)

    parts = []
    for y in years:
        path = f"csv/{y}_all.csv.gz"
        if os.path.exists(path):
            print(f"[READ] {path}")
            parts.append(read_all_standardized(path))
        else:
            print(f"[WARN] not found: {path}")
    if not parts:
        raise RuntimeError("ALL 파일이 없습니다.")

    all_df = pd.concat(parts, ignore_index=True)
    print(f"[INFO] ALL merged shape: {all_df.shape}")
    print("[INFO] Columns(sample):", list(all_df.columns)[:20])

    dfh = all_df.copy()

    # 파생: 성별/연령
    if "sex" in dfh.columns:
        dfh["sex_female"] = (dfh["sex"] == 2).astype("Int64")
    if "age" in dfh.columns:
        dfh["age"] = pd.to_numeric(dfh["age"], errors="coerce")

    # Outcomes ----------------------------------------------------------------
    # 비만
    if "BMI" in dfh.columns:
        dfh["out_obesity"] = (dfh["BMI"] >= float(obesity_cut)).astype("Int64")

    # 고혈압: SBP/DBP 기준 + 약물(있으면 포함)
    if {"SBP", "DBP"}.issubset(dfh.columns):
        base_htn = (dfh["SBP"] >= 140) | (dfh["DBP"] >= 90)
        if "htn_med" in dfh.columns:
            dfh["out_hypertension"] = (base_htn | (dfh["htn_med"] == 1)).astype("Int64")
        else:
            dfh["out_hypertension"] = base_htn.astype("Int64")

    # 당뇨: HbA1c 기준 + 약물(있으면 포함)
    if "HbA1c" in dfh.columns:
        base_dm = (dfh["HbA1c"] >= 6.5)
        if "dm_med" in dfh.columns:
            dfh["out_diabetes"] = (base_dm | (dfh["dm_med"] == 1)).astype("Int64")
        else:
            dfh["out_diabetes"] = base_dm.astype("Int64")

    # 최종 컬럼 구성(존재하는 것만 선택)
    keep = [
        "ID", "year", "sex_female", "age",
        "out_obesity", "out_hypertension", "out_diabetes",
        "incm", "edu", "wt_tot", "psu", "kstrata",
        # 참고용으로 남겨두면 이후 기술통계에 유용
        "smoking", "alcohol", "kcal", "BMI", "SBP", "DBP", "HbA1c"
    ]
    final = [c for c in keep if c in dfh.columns]
    dfh = dfh[final].dropna(subset=["age"])

    # 간단 점검 로그
    for col in ["out_obesity", "out_hypertension", "out_diabetes"]:
        if col in dfh.columns:
            s = pd.to_numeric(dfh[col], errors="coerce").fillna(0).astype(int).sum()
            print(f"[CHECK] {col} sum = {int(s)}")
        else:
            print(f"[CHECK] {col} not present")

    outp = "csv/health_2014_2017.csv.gz"
    dfh.to_csv(outp, index=False, compression="gzip")
    print(f"[Saved] {outp} | shape={dfh.shape}")
    print("Health columns:", list(dfh.columns))
    return dfh

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2014, 2015, 2016, 2017])
    parser.add_argument("--obesity-cut", type=float, default=25.0)  # 한국 기준 기본 25
    args = parser.parse_args()

    build_health(args.years, args.obesity_cut)
