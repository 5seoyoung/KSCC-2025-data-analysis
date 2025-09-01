# KNHANES Accelerometry × Metabolic Risk (2014–2017)

가속도계 기반 신체활동(일평균 MVPA, 좌식비율)과 비만·고혈압·당뇨병 등 대사위험의 연관성을 국민건강영양조사(KNHANES) 2014–2017 자료로 분석한 재현 가능한 파이프라인입니다.

전처리 → 기술통계 → 가중 로지스틱 → 스플라인(RCS) → XGBoost/SHAP까지 스크립트 1–6단계로 구성됩니다.

## 데이터 안내

이 저장소에는 원자료가 포함되어 있지 않습니다. 용량 및 이용조건 문제로 사용자가 직접 다운로드해야 합니다(아래 "데이터 준비" 참고).

분석은 최종 축약본(`csv/analysis_ready_expanded.csv.gz`) 기준으로 수행됩니다.

## 폴더/파일 구조

```
src/
 ├─ 00_build_health.py           # 건강/검진/설문/영양 원자료 정리
 ├─ 01_summarize_pam.py          # 가속도계(PAM) 요약지표 산출
 ├─ 02_build_analysis_ready.py   # 개인단위 머지 → 최종분석용 축약본 생성
 ├─ 03_descriptives.py           # 기술통계/그룹비교 표
 ├─ 04_models.py                 # 가중 로지스틱, OR 요약, 포리스트플롯
 ├─ 05_spline.py                 # RCS 곡선/예측치 (MVPA, 좌식비율)
 └─ 06_ml_shap.py                # XGBoost, AUC, SHAP(bar/beeswarm/dependence)
convert_and_qc.ipynb             # (선택) 수기 점검/시각화 노트북
csv/                              # 사용자 준비: 최종 축약본 or 원자료
out/                              # 모든 결과 표/그림이 생성되는 폴더
```

## 빠른 시작 (TL;DR)

```bash
# 1) 가상환경 권장 (예: Python 3.11)
python -m venv .venv && source .venv/bin/activate

# 2) 필수 패키지
pip install pandas numpy scipy statsmodels scikit-learn xgboost shap matplotlib

# 3) (Mac/Apple Silicon만) XGBoost OpenMP 의존성
# brew install libomp
# export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH}"

# 4) 최종 축약본이 준비된 경우(권장): csv/analysis_ready_expanded.csv.gz 를 배치
# 없으면 아래 '데이터 준비' 참고하여 00~02 단계 실행

# 5) 결과 생성
python src/03_descriptives.py
python src/04_models.py
python src/05_spline.py
python src/06_ml_shap.py --data csv/analysis_ready_expanded.csv.gz --max-n 5000 --seed 42
```

### 생성물

모든 결과는 `out/`에 저장됩니다:

**표:**
- `desc_continuous.csv`, `desc_categorical.csv`, `group_compare.csv`
- `summary_or_exposures.csv`, `spline_pred_*.csv`

**그림:**
- `forest_weighted_all.png`, `spline_mvpa_min_day.png`, `spline_sed_ratio.png`
- `shap_bar.png`, `shap_beeswarm.png`, `shap_dependence.png`

## 데이터 준비

### 원자료 다운로드 (사용자 직접)

- **출처:** 국민건강영양조사(KNHANES) 2014–2017 (검진/건강설문/영양/가속도계)
- **참고:** 제공 포맷과 파일명은 연도별로 상이할 수 있습니다

### 선택 경로 A: 원자료 → 최종 축약본 생성

원파일 경로를 스크립트에서 읽도록 정리한 후 아래 순서로 실행:

```bash
python src/00_build_health.py
python src/01_summarize_pam.py
python src/02_build_analysis_ready.py
```

완료되면 `csv/analysis_ready_expanded.csv.gz`가 생성됩니다.

### 선택 경로 B: 축약본 바로 사용 (권장)

이미 생성된 `csv/analysis_ready_expanded.csv.gz`를 `csv/`에 두고 03–06단계만 실행하세요.

**개인정보 보호:** ID는 분석용 임의 식별자로 치환됩니다.

## 재현 파이프라인 (상세)

### 1단계. 전처리/병합

- `00_build_health.py`: 검진/설문/영양 핵심 변수 추출·표준화
- `01_summarize_pam.py`: 가속도계 원신호 요약(일평균 MVPA 분, 좌식비율, 착용일수·시간 QC)
- `02_build_analysis_ready.py`: 개인단위 병합 및 포함/제외 기준 적용(성인, ≥4일, 일 착용≥600분)

### 2단계. 기술통계/그룹비교

```bash
python src/03_descriptives.py
```

**출력:** `out/desc_continuous.csv`, `out/desc_categorical.csv`, `out/group_compare.csv`

### 3단계. 가중 로지스틱 & 포리스트

```bash
python src/04_models.py
```

**출력:** `out/summary_or_exposures.csv`, `out/forest_weighted_all.png`

### 4단계. 스플라인(RCS)

```bash
python src/05_spline.py
```

**출력:**
- `out/spline_mvpa_min_day.png`, `out/spline_sed_ratio.png`
- `out/spline_pred_mvpa_min_day_*.csv`, `out/spline_pred_sed_ratio_*.csv`

### 5단계. XGBoost/SHAP

```bash
python src/06_ml_shap.py --data csv/analysis_ready_expanded.csv.gz --max-n 5000 --seed 42
```

**출력:**
- 콘솔: AUC(검증)
- 그림: `out/shap_bar.png`, `out/shap_beeswarm.png`, `out/shap_dependence.png`

## 원고 매핑 (표/그림)

| 원고 | 파일 |
|------|------|
| **표 1** 연속형 특성 | `out/desc_continuous.csv` |
| **표 2** 범주형 특성 | `out/desc_categorical.csv` |
| **표 3** 그룹 비교 | `out/group_compare.csv` |
| **표 4** OR 요약 | `out/summary_or_exposures.csv` |
| **그림 1** 포리스트 | `out/forest_weighted_all.png` |
| **그림 2** RCS(MVPA) | `out/spline_mvpa_min_day.png` + `spline_pred_mvpa_min_day_*.csv` |
| **그림 3** RCS(Sedentary) | `out/spline_sed_ratio.png` + `spline_pred_sed_ratio_*.csv` |
| **그림 4–6** SHAP | `out/shap_bar.png`, `out/shap_beeswarm.png`, `out/shap_dependence.png` |

## 환경/호환성 메모

- **Python:** 3.11 권장
- **Mac(Apple Silicon):**
  - XGBoost 실행 전 `brew install libomp` 후 `DYLD_LIBRARY_PATH` 설정 필요
  - SHAP는 numba 제약으로 NumPy 2.2.x 이하 권장
- **재현성:** 모든 스크립트는 `--seed 42`로 재현성을 유지
