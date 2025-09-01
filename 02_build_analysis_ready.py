"""
02_build_analysis_ready.py

- KNHANES PAM minute-level CSV(.gz)들을 읽어
  (ID, year, day, minute, inten[, hour]) 표준 스키마로 정규화
- 일(day) 단위 요약 -> 개인(person) 단위 요약 생성
- health_2014_2017.csv.gz 와 병합하여 분석용 데이터 생성

Usage:
  python src/02_build_analysis_ready.py \
    --mvpa-cut 2020 --sed-cut 100 --min-wear 600 --min-days 4

출력:
  csv/activity_day_2014_2017.csv.gz       (일 단위 요약)
  csv/activity_person_2014_2017.csv.gz    (개인 단위 요약)
  csv/analysis_ready_expanded.csv.gz      (건강+활동 병합, 파생변수 포함)
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 200)

# ---------- 경로 설정 ----------
YEARS = [2014, 2015, 2016, 2017]
PATH_PAM = {
    2014: "csv/2014_pam.csv.gz",
    2015: "csv/2015_pam.csv.gz",
    2016: "csv/2016_pam.csv.gz",
    2017: "csv/2017_pam.csv.gz",
}
HEALTH_PATH = "csv/health_2014_2017.csv.gz"

Path("csv").mkdir(exist_ok=True)
Path("out").mkdir(exist_ok=True)


# ---------- 내부 유틸 ----------
def _first_series(d: pd.DataFrame, candidates):
    """
    candidates 중 존재하는 첫 열을 1-D Series로 반환 (없으면 None).
    동일 이름 중복으로 DataFrame이 나오면 첫 컬럼만 사용.
    """
    for c in candidates:
        if c in d.columns:
            col = d[c]
            if isinstance(col, pd.DataFrame):  # 중복 열명 방어
                col = col.iloc[:, 0]
            return col
    return None


# ---------- 1) 분(minute) 단위 로더 ----------
def read_pam_minute(path_csv_gz: str) -> pd.DataFrame:
    """
    PAM minute-level 파일을 표준 스키마로 정규화:
    [ID, year, day, minute, inten, (hour optional)]
    """
    d = pd.read_csv(path_csv_gz, low_memory=False)
    # 완전 중복/중복명 제거
    d.columns = [str(c).strip() for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()].copy()

    # 소문자 매핑 헬퍼
    lower2orig = {c.lower(): c for c in d.columns}

    id_s   = _first_series(d, [lower2orig.get("id", "id")])
    yr_s   = _first_series(d, [lower2orig.get("year", "year")])
    day_s  = _first_series(d, [
        lower2orig.get("day", "day"),
        lower2orig.get("paxday", "PAXDAY"),
        lower2orig.get("paxn", "PAXN"),
        lower2orig.get("dayno", "dayno"),
    ])
    hour_s = _first_series(d, [
        lower2orig.get("hour", "hour"),
        lower2orig.get("paxhour", "PAXHOUR"),
    ])
    min_s  = _first_series(d, [
        lower2orig.get("minute", "minute"),
        lower2orig.get("paxminut", "PAXMINUT"),
        lower2orig.get("paxminute", "PAXMINUTE"),
    ])
    inten_s = _first_series(d, [
        lower2orig.get("paxinten", "PAXINTEN"),
        lower2orig.get("inten", "inten"),
        lower2orig.get("counts", "counts"),
    ])

    req = {"ID": id_s, "year": yr_s, "day": day_s, "minute": min_s, "inten": inten_s}
    missing = [k for k, v in req.items() if v is None]
    if missing:
        raise ValueError(f"[{path_csv_gz}] required columns missing after standardization: {missing}")

    std = pd.DataFrame({
        "ID": req["ID"].astype(str),
        "year": pd.to_numeric(req["year"], errors="coerce").astype("Int64"),
        "day": pd.to_numeric(req["day"], errors="coerce").astype("Int64"),
        "minute": pd.to_numeric(req["minute"], errors="coerce").astype("Int64"),
        "inten": pd.to_numeric(req["inten"], errors="coerce")
    })
    if hour_s is not None:
        std["hour"] = pd.to_numeric(hour_s, errors="coerce").astype("Int64")

    # 기본 품질 로그
    print(f"[READ minute] {path_csv_gz} -> shape={std.shape} | cols={list(std.columns)}")
    return std


# ---------- 2) 일(day) 단위 요약 ----------
def summarize_day(pam_min: pd.DataFrame, mvpa_cut: float, sed_cut: float, min_wear: int) -> pd.DataFrame:
    """
    분 단위로부터 일 단위 요약:
      worn_min_day: 유효 착용 분(inten notnull)
      mvpa_min_day: inten >= mvpa_cut 분 수
      sed_ratio:    inten < sed_cut 비율 (sedentary / worn)
    day-level 필터: worn_min_day >= min_wear
    """
    # 유효 착용 플래그
    pam_min = pam_min.copy()
    pam_min["is_worn"] = pam_min["inten"].notna().astype(int)
    pam_min["is_mvpa"] = (pam_min["inten"] >= mvpa_cut).astype(int)
    pam_min["is_sed"]  = (pam_min["inten"] < sed_cut).astype(int)

    grp = pam_min.groupby(["ID", "year", "day"], as_index=False).agg(
        worn_min_day=("is_worn", "sum"),
        mvpa_min_day=("is_mvpa", "sum"),
        sed_min_day=("is_sed", "sum")
    )
    grp["sed_ratio"] = np.where(grp["worn_min_day"] > 0,
                                grp["sed_min_day"] / grp["worn_min_day"],
                                np.nan)

    # 일 단위 착용 기준 적용
    before = grp.shape[0]
    grp = grp.loc[grp["worn_min_day"] >= min_wear].copy()
    after = grp.shape[0]
    print(f"[DAY] kept {after}/{before} days (wear ≥ {min_wear} min)")

    # 저장
    out_day = "csv/activity_day_2014_2017.csv.gz"
    grp.to_csv(out_day, index=False, compression="gzip")
    print(f"[SAVE] day-level -> {out_day} | rows={grp.shape[0]}")
    return grp


# ---------- 3) 개인(person) 단위 요약 ----------
def summarize_person(day_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """
    day_df를 개인 단위로 요약:
      n_days: 유지된 분석가능 일수
      worn_min_day, mvpa_min_day, sed_ratio: 일평균
    person-level 필터: n_days ≥ min_days
    """
    agg = (day_df
           .groupby(["ID", "year"], as_index=False)
           .agg(n_days=("day", "count"),
                worn_min_day=("worn_min_day", "mean"),
                mvpa_min_day=("mvpa_min_day", "mean"),
                sed_ratio=("sed_ratio", "mean")))

    before = agg.shape[0]
    agg = agg.loc[agg["n_days"] >= min_days].copy()
    after = agg.shape[0]
    print(f"[PERSON] kept {after}/{before} persons (days ≥ {min_days})")

    out_person = "csv/activity_person_2014_2017.csv.gz"
    agg.to_csv(out_person, index=False, compression="gzip")
    print(f"[SAVE] person-level -> {out_person} | rows={agg.shape[0]}")
    return agg


# ---------- 4) 최종 병합 및 파생 ----------
def build_analysis_ready(mvpa_cut: float, sed_cut: float, min_wear: int, min_days: int) -> None:
    # (a) minute 파일들 읽기
    parts = []
    for y in YEARS:
        p = PATH_PAM[y]
        if os.path.exists(p):
            parts.append(read_pam_minute(p))
        else:
            print(f"[PAM] not found: {p}")
    assert parts, "No PAM minute files found."
    pam_min = pd.concat(parts, ignore_index=True)
    pam_min = pam_min.loc[:, ~pam_min.columns.duplicated()].copy()

    # (b) day/person 요약
    day_df = summarize_day(pam_min, mvpa_cut, sed_cut, min_wear=min_wear)
    person_df = summarize_person(day_df, min_days=min_days)

    # (c) health 불러오기
    assert os.path.exists(HEALTH_PATH), f"health file not found: {HEALTH_PATH}"
    hlth = pd.read_csv(HEALTH_PATH, low_memory=False)
    # 표준 키 타입 맞추기
    for d in (person_df, hlth):
        if "year" in d.columns:
            d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
        if "ID" in d.columns:
            d["ID"] = d["ID"].astype(str)

    # (d) 병합
    df = person_df.merge(hlth, on=["ID", "year"], how="inner")
    print(f"[MERGE] analysis-ready shape: {df.shape}")

    # (e) 파생
    if "mvpa_min_day" in df.columns:
        df["mvpa10"] = df["mvpa_min_day"] / 10.0
    if "sed_ratio" in df.columns:
        df["sed10"] = df["sed_ratio"] * 10.0

    # (f) 가중치/설계 없으면 폴백
    if "wt_tot" not in df.columns:
        print("[HOTFIX] wt_tot missing -> set to 1.0")
        df["wt_tot"] = 1.0
    if not {"psu", "kstrata"}.issubset(df.columns):
        print("[HOTFIX] psu/kstrata missing -> design-based bootstrap not available")

    # (g) 저장
    outp = "csv/analysis_ready_expanded.csv.gz"
    df.to_csv(outp, index=False, compression="gzip")
    print(f"[SAVE] analysis-ready -> {outp} | rows={df.shape[0]}")
    print("Final columns (head):", list(df.columns)[:20])


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Build analysis-ready data from PAM minute files + health table.")
    ap.add_argument("--mvpa-cut", type=float, default=2020,
                    help="counts/min threshold for MVPA (default: 2020)")
    ap.add_argument("--sed-cut", type=float, default=100,
                    help="counts/min threshold for sedentary (default: 100)")
    ap.add_argument("--min-wear", type=int, default=600,
                    help="valid day wear-time threshold in minutes (default: 600)")
    ap.add_argument("--min-days", type=int, default=4,
                    help="min number of valid days per person (default: 4)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Working dir:", os.getcwd())
    print(f"Params -> mvpa_cut={args.mvpa_cut}, sed_cut={args.sed_cut}, min_wear={args.min_wear}, min_days={args.min_days}")
    build_analysis_ready(
        mvpa_cut=args.mvpa_cut,
        sed_cut=args.sed_cut,
        min_wear=args.min_wear,
        min_days=args.min_days,
    )
