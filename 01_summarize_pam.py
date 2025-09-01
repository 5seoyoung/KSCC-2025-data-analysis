
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
warnings.simplefilter("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------
YEARS = [2014, 2015, 2016, 2017]
PATH_ALLS = {
    2014: "csv/2014_all.csv.gz",
    2015: "csv/2015_all.csv.gz",
    2016: "csv/2016_all.csv.gz",
    2017: "csv/2017_all.csv.gz",
}
PATH_PAM = {
    2014: "csv/2014_pam.csv.gz",
    2015: "csv/2015_pam.csv.gz",
    2016: "csv/2016_pam.csv.gz",
    2017: "csv/2017_pam.csv.gz",
}

Path("csv").mkdir(exist_ok=True)
Path("out").mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# 1) HEALTH (ALL) loader with standardized columns
# --------------------------------------------------------------------------------------
MAP_ALL = {
    "ID"      : "ID",
    "year"    : "year",
    "sex"     : "sex",
    "age"     : "age",
    "BMI"     : "BMI",
    "SBP"     : "SBP",
    "DBP"     : "DBP",
    "HbA1c"   : "HBA1C",
    "htn_med" : "htn_med",
    "dm_med"  : "dm_med",
    "smoking" : "smoking",
    "alcohol" : "alcohol",
    "kcal"    : "kcal",
    "incm"    : "incm",
    "edu"     : "edu",
    "wt_tot"  : "wt_tot",
    "psu"     : "psu",
    "kstrata" : "kstrata",
}

def read_all_standardized(path_csv_gz: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv_gz, low_memory=False)
    df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
    # Build reverse map in a case-insensitive way
    rev = {}
    for std_name, real_name in MAP_ALL.items():
        hits = [c for c in df.columns if str(c).lower() == str(real_name).lower()]
        if hits:
            rev[hits[0]] = std_name
        elif real_name in df.columns:
            rev[real_name] = std_name
    df = df.rename(columns=rev)
    if not {"ID","year"}.issubset(df.columns):
        raise ValueError(f"[{path_csv_gz}] ID/year 없음 → MAP_ALL 확인 필요")
    keep = [c for c in MAP_ALL.keys() if c in df.columns]
    df = df[keep].copy()
    for c in df.columns:
        if c != "ID":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ID"] = df["ID"].astype(str)
    return df

def build_health_table(path_alls_map: Dict[int, str] = PATH_ALLS) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for y, p in path_alls_map.items():
        if os.path.exists(p):
            print(f"[ALL] using {p}")
            d = read_all_standardized(p)
            parts.append(d)
        else:
            print(f"[WARN] ALL not found, skip: {p}")
    if not parts:
        raise RuntimeError("No ALL files to build health table.")
    all_df = pd.concat(parts, ignore_index=True)
    print("ALL merged shape:", all_df.shape)
    print("ALL columns (head):", list(all_df.columns)[:12])

    dfh = all_df.copy()
    if "sex" in dfh.columns:
        dfh["sex_female"] = (dfh["sex"] == 2).astype("Int64")
    if "age" in dfh.columns:
        dfh["age"] = pd.to_numeric(dfh["age"], errors="coerce")

    # Outcomes
    if "BMI" in dfh.columns:
        dfh["out_obesity"] = (dfh["BMI"] >= 30).astype("Int64")
    else:
        dfh["out_obesity"] = pd.NA

    dfh["out_hypertension"] = (
        (dfh["SBP"] >= 140) |
        (dfh["DBP"] >= 90)  |
        (dfh.get("htn_med", 0).fillna(0) == 1)
    ).astype("Int64") if {"SBP","DBP"}.issubset(dfh.columns) else pd.Series([pd.NA]*len(dfh), dtype="Int64")

    dfh["out_diabetes"] = (
        (dfh["HbA1c"] >= 6.5) | (dfh.get("dm_med", 0).fillna(0) == 1)
    ).astype("Int64") if ("HbA1c" in dfh.columns) else pd.Series([pd.NA]*len(dfh), dtype="Int64")

    # Behaviors
    dfh["smoker"]       = (dfh.get("smoking", 0).fillna(0) > 0).astype("Int64") if "smoking" in dfh.columns else pd.Series([pd.NA]*len(dfh), dtype="Int64")
    dfh["alcohol_freq"] = pd.to_numeric(dfh.get("alcohol", pd.NA), errors="coerce")

    keep = ["ID","year","sex_female","age","smoker","alcohol_freq",
            "out_obesity","out_hypertension","out_diabetes",
            "incm","edu","wt_tot","psu","kstrata"]
    dfh = dfh[[c for c in keep if c in dfh.columns]]
    outp = "csv/health_2014_2017.csv.gz"
    dfh.to_csv(outp, index=False, compression="gzip")
    print(f"[Saved] {outp} | shape={dfh.shape}")
    print("Health columns:", list(dfh.columns))
    return dfh

# --------------------------------------------------------------------------------------
# 2) PAM (activity) person-level builder — forgiving version
# --------------------------------------------------------------------------------------
PAM_ACCEPT_COLS = {
    "ID": ["id", "ID"],
    "year": ["year", "YEAR"],
    "n_days": ["n_days", "ndays", "valid_days"],
    "worn_min_day": ["worn_min_day", "wear_min_day", "wear_time_min_day"],
    "mvpa_min_day": ["mvpa_min_day", "mvpa_minute_day", "mvpa_min"],
    "sed_ratio": ["sed_ratio", "sedentary_ratio", "sedentary_prop", "sed_ratio_day"],
}

def _rename_loose(df: pd.DataFrame, mapping: Dict[str, list]) -> pd.DataFrame:
    ren = {}
    cols_lower = {str(c).lower(): c for c in df.columns}
    for std, cand in mapping.items():
        for name in cand:
            if name.lower() in cols_lower:
                ren[cols_lower[name.lower()]] = std
                break
    return df.rename(columns=ren)

def build_activity_person(path_pam_map: Dict[int, str]) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for y, p in path_pam_map.items():
        if not os.path.exists(p):
            print(f"[PAM] 없음: {p}")
            continue
        try:
            d = pd.read_csv(p, low_memory=False)
        except Exception as e:
            print(f"[PAM] read fail: {p} | {e}")
            continue

        d = _rename_loose(d, PAM_ACCEPT_COLS)

        # Require ID/year minimally
        if not {"ID","year"}.issubset(d.columns):
            print(f"[PAM] {p} → ID/year 누락, skip")
            continue

        # If minute-level blew up (ID/year only), deduplicate to unique pairs
        keep_cols = [c for c in ["ID","year","n_days","worn_min_day","mvpa_min_day","sed_ratio"] if c in d.columns]
        d = d[keep_cols].copy()

        # Coerce types
        d["ID"] = d["ID"].astype(str)
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")

        # If no activity metrics present, create NaNs and deduplicate (ID, year)
        act_cols = [c for c in ["n_days","worn_min_day","mvpa_min_day","sed_ratio"] if c in d.columns]
        if len(act_cols) == 0:
            print(f"[WARN] {p} mvpa/sed 없음 → 빈 변수 생성 + 중복 제거")
            d = d.drop_duplicates(["ID","year"]).copy()
            for c in ["n_days","worn_min_day","mvpa_min_day","sed_ratio"]:
                if c not in d.columns:
                    d[c] = np.nan
        else:
            # If multiple rows per person, keep the last non-null by (ID,year)
            d = d.sort_values(by=list(d.columns)).drop_duplicates(["ID","year"], keep="last")

        frames.append(d)

    if not frames:
        print("[PAM] no frames collected")
        return None

    act = pd.concat(frames, ignore_index=True)
    print("PAM person (dedup) shape:", act.shape)
    print("PAM columns:", list(act.columns))
    # Quick diagnostics
    if "worn_min_day" in act.columns:
        uniq = act["worn_min_day"].round(1).dropna().nunique()
        if uniq <= 2:
            print(f"[Note] worn_min_day 분산 낮음 (unique={uniq})")
    return act

# --------------------------------------------------------------------------------------
# 3) Build analysis-ready table
# --------------------------------------------------------------------------------------
def build_analysis_ready(path_alls_map: Dict[int,str] = PATH_ALLS,
                         path_pam_map: Dict[int,str] = PATH_PAM,
                         out_csv: str = "csv/analysis_ready_expanded.csv.gz") -> pd.DataFrame:
    # Health
    if os.path.exists("csv/health_2014_2017.csv.gz"):
        hlth = pd.read_csv("csv/health_2014_2017.csv.gz", low_memory=False)
    else:
        hlth = build_health_table(path_alls_map)

    # Activity (forgiving)
    act = build_activity_person(path_pam_map)

    # Harmonize keys
    for d in [hlth, act] if act is not None else [hlth]:
        if "id" in d.columns and "ID" not in d.columns:
            d.rename(columns={"id": "ID"}, inplace=True)
        d["ID"] = d["ID"].astype(str)
        if "year" in d.columns:
            d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")

    # Merge (inner if act exists, else health only)
    if act is not None:
        df = act.merge(hlth, on=["ID","year"], how="inner")
        # If activity metrics absent, columns will be NaN but fine
    else:
        print("[Note] PAM 요약 없음 → health만 사용")
        df = hlth.copy()

    # Derived exposures
    if "mvpa_min_day" in df.columns:
        df["mvpa10"] = pd.to_numeric(df["mvpa_min_day"], errors="coerce") / 10.0
    if "sed_ratio" in df.columns:
        df["sed10"]  = pd.to_numeric(df["sed_ratio"], errors="coerce") * 10.0

    # Weights / design fallbacks
    if "wt_tot" not in df.columns:
        print("[HOTFIX] 가중치 미존재 → wt_tot=1.0 설정")
        df["wt_tot"] = 1.0
    if not {"psu","kstrata"}.issubset(df.columns):
        print("[HOTFIX] psu/kstrata 없음 → 설계부트 불가(후속분석 폴백)")

    # Final de-dup (safety): one row per (ID,year)
    df = df.sort_values(by=list(df.columns)).drop_duplicates(["ID","year"], keep="last")

    # Save
    df.to_csv(out_csv, index=False, compression="gzip")
    print(f"[Saved] {out_csv} | shape={df.shape}")
    print("Final columns (head):", list(df.columns)[:20])
    return df

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    df_final = build_analysis_ready(PATH_ALLS, PATH_PAM)
    print(df_final.head())
