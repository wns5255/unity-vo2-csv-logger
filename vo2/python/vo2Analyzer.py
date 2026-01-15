# -*- coding: utf-8 -*-
# vo2Analyzer.py
# Polar CSV + Analyzer CSV → 1s 리샘플/정렬 → (옵션) 지연 τ 추정 → 병합
# → 시각화 → 학습(보정: Linear/Huber/Ridge) → 결과 저장
#
# pip install pandas numpy matplotlib scikit-learn

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ======================= 기본 설정(경로/옵션) =======================
DEFAULT_POLAR_PATH    = r"C:\Users\wns5255\Desktop\TWIN\polar.csv"
DEFAULT_ANALYZER_PATH = r"C:\Users\wns5255\Desktop\TWIN\analyzer.csv"

# analyzer.csv 2행이 단위라인이면 True → skiprows=[1]
ANALYZER_HAS_UNIT_ROW = True

# Analyzer VO2 컬럼 후보 (파일마다 이름이 다를 수 있음)
ANALYZER_VO2_CANDIDATES = ["VO2", "vo2", "VO2_ml_per_min", "vo2_ml_min"]

# 리샘플/평활 설정
RESAMPLE_SEC = "1s"   # '1S' 말고 '1s' 사용 (FutureWarning 회피)
SMOOTH_SEC   = 20     # Analyzer VO2 이동평균(초) — 노이즈 완화

# 지연(τ) 자동 추정 범위(초). 0이면 탐색 안 함
LAG_SEARCH_RANGE = 60  # -60~+60초에서 탐색

# 학습에 기본으로 쓸 피처(Polar VO2는 필수)
BASE_FEATURES = ["vo2_polar_mlmin"]
# ================================================================


def mmss_to_sec(x: str) -> int:
    m, s = str(x).split(":")
    return int(m) * 60 + int(s)

def ccc(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mu1, mu2 = np.mean(y_true), np.mean(y_pred)
    s1, s2  = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    cov     = np.cov(y_true, y_pred, ddof=1)[0,1]
    return (2 * cov) / (s1 + s2 + (mu1 - mu2)**2 + 1e-12)


def load_polar(polar_path: str) -> pd.DataFrame:
    if not os.path.exists(polar_path):
        raise FileNotFoundError(f"Polar CSV가 없습니다: {polar_path}")
    dfp = pd.read_csv(polar_path)

    # 시간 열: epochMs 우선, 없으면 localTime
    if "epochMs" in dfp.columns:
        dfp["ts"] = pd.to_datetime(pd.to_numeric(dfp["epochMs"], errors="coerce"), unit="ms")
    elif "localTime" in dfp.columns:
        dfp["ts"] = pd.to_datetime(dfp["localTime"])
    else:
        raise ValueError("polar.csv에 epochMs 또는 localTime 컬럼이 필요합니다.")

    # VO2(L/min) → mL/min
    if "vo2Abs_Lmin" not in dfp.columns:
        raise ValueError("polar.csv에 vo2Abs_Lmin 컬럼이 필요합니다.")
    dfp["vo2_polar_mlmin"] = pd.to_numeric(dfp["vo2Abs_Lmin"], errors="coerce") * 1000.0

    # 숫자만 1초 리샘플(평균)
    num_cols = dfp.select_dtypes(include=[np.number]).columns.tolist()
    dfp = dfp.set_index("ts")
    dfp_num = dfp[num_cols].resample(RESAMPLE_SEC).mean()
    # 필요한 컬럼만 유지 (있으면 사용)
    keep_cols = [c for c in ["hr", "avgHr", "hrr", "vo2_polar_mlmin"] if c in dfp_num.columns]
    dfp_num = dfp_num[keep_cols].interpolate().dropna(how="all")
    if dfp_num.empty:
        raise RuntimeError("Polar 리샘플 결과가 비었습니다. 입력 CSV/컬럼을 확인하세요.")
    return dfp_num


def load_analyzer(analyzer_path: str, polar_start_ts: pd.Timestamp) -> pd.DataFrame:
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Analyzer CSV가 없습니다: {analyzer_path}")
    skip = [1] if ANALYZER_HAS_UNIT_ROW else None
    dfa_raw = pd.read_csv(analyzer_path, skiprows=skip)

    if "t" not in dfa_raw.columns:
        raise ValueError("analyzer.csv에 t(mm:ss) 컬럼이 필요합니다.")

    # VO2 컬럼 찾기
    vo2_col = None
    for cand in ANALYZER_VO2_CANDIDATES:
        if cand in dfa_raw.columns:
            vo2_col = cand
            break
    if vo2_col is None:
        raise ValueError(f"Analyzer VO2 컬럼을 찾을 수 없습니다. 후보: {ANALYZER_VO2_CANDIDATES}")

    dfa = dfa_raw.copy()
    dfa["t_sec"] = dfa["t"].apply(mmss_to_sec)
    dfa["ts"] = polar_start_ts + dfa["t_sec"].apply(lambda s: timedelta(seconds=int(s)))

    # 1초 리샘플 + 보간 + 이동평균
    dfa = (dfa.set_index("ts")[[vo2_col]]
             .apply(pd.to_numeric, errors="coerce")
             .resample(RESAMPLE_SEC).mean()
             .interpolate()
             .rolling(SMOOTH_SEC, min_periods=1, center=True).mean()
             .rename(columns={vo2_col: "vo2_true_mlmin"}))
    if dfa.empty:
        raise RuntimeError("Analyzer 리샘플 결과가 비었습니다. 입력 CSV/컬럼을 확인하세요.")
    return dfa


def estimate_best_lag(dfp_num: pd.DataFrame, dfa: pd.DataFrame, search_range: int) -> int:
    if search_range <= 0:
        return 0

    def corr_at_shift(shift_sec: int):
        shifted = dfa.copy()
        shifted.index = shifted.index + pd.Timedelta(seconds=shift_sec)
        tmp = pd.merge_asof(
            dfp_num[["vo2_polar_mlmin"]].sort_index(),
            shifted[["vo2_true_mlmin"]].sort_index(),
            left_index=True, right_index=True,
            direction="nearest", tolerance=pd.Timedelta(seconds=1)
        ).dropna()
        if len(tmp) < 10:
            return np.nan
        return np.corrcoef(tmp["vo2_polar_mlmin"], tmp["vo2_true_mlmin"])[0,1]

    candidates = range(-search_range, search_range+1)
    scores = {s: corr_at_shift(s) for s in candidates}
    best_tau = max(scores, key=lambda k: (-np.inf if pd.isna(scores[k]) else scores[k]))
    print(f"[INFO] Best lag τ = {best_tau} sec (corr={scores[best_tau]:.3f})")
    return int(best_tau)


def run_training(merged: pd.DataFrame, user_info: dict):
    # 사용자 정보(체중, vo2max, 안정시HR, 키, 나이, 성별) 추가
    # sex: 남=1, 여=0 로 인코딩. 문자열 입력도 허용.
    def sex_to_num(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ["m", "male", "man", "남", "남자"]: return 1
            if s in ["f", "female", "woman", "여", "여자"]: return 0
        try:
            return int(x)
        except:
            return 1  # 기본값: 남
    user_info = user_info.copy()
    user_info["sex"] = sex_to_num(user_info.get("sex", 1))

    for k, v in user_info.items():
        merged[k] = v

    # 피처 구성: BASE_FEATURES + (hr 있으면 추가) + 사용자 정보
    feat_cols = BASE_FEATURES.copy()
    if "hr" in merged.columns and "hr" not in feat_cols:
        feat_cols.append("hr")
    elif "avgHr" in merged.columns and "hr" not in feat_cols:
        # hr 없고 avgHr만 있으면 대체 사용
        merged["hr"] = merged["avgHr"]
        feat_cols.append("hr")

    # 사용자 특성 추가
    for add in ["weight", "vo2max", "resting_hr", "height", "age", "sex"]:
        if add not in merged.columns:
            raise ValueError(f"사용자 입력 누락: {add}")
        feat_cols.append(add)

    # X, y
    X = merged[feat_cols].values
    y = merged["vo2_true_mlmin"].values

    def report(name, pred):
        return {
            "name": name,
            "R2": float(r2_score(y, pred)),
            "MAPE_pct": float(mean_absolute_percentage_error(y, pred) * 100),
            "CCC": float(ccc(y, pred)),
        }

    # Linear
    lin = LinearRegression().fit(X, y)
    pred_lin = lin.predict(X)
    res_lin  = report("Linear", pred_lin)

    # Huber (robust)
    hub = HuberRegressor(alpha=1e-3).fit(X, y)
    pred_hub = hub.predict(X)
    res_hub  = report("Huber", pred_hub)

    # Ridge
    rid = Ridge(alpha=1.0).fit(X, y)
    pred_rid = rid.predict(X)
    res_rid  = report("Ridge", pred_rid)

    print("\n=== Calibration Results (with user features) ===")
    for res in [res_lin, res_hub, res_rid]:
        print(f"{res['name']:>6s} | R²={res['R2']:.3f}  MAPE={res['MAPE_pct']:.2f}%  CCC={res['CCC']:.3f}")

    # Huber를 기본 선택
    best_name = "Huber"
    best_pred = pred_hub
    best_coef = hub.coef_.tolist()
    best_intercept = float(hub.intercept_)

    return {
        "feat_cols": feat_cols,
        "best_name": best_name,
        "best_pred": best_pred,
        "lin": (lin.coef_.tolist(), float(lin.intercept_)),
        "hub": (best_coef, best_intercept),
        "rid": (rid.coef_.tolist(), float(rid.intercept_)),
    }


def main():
    parser = argparse.ArgumentParser(description="Polar vs Analyzer VO₂ 보정/학습")
    parser.add_argument("--polar", default=DEFAULT_POLAR_PATH, help="Polar CSV 경로")
    parser.add_argument("--analyzer", default=DEFAULT_ANALYZER_PATH, help="Analyzer CSV 경로")
    # 사용자 입력
    parser.add_argument("--weight", type=float, required=True, help="체중 kg")
    parser.add_argument("--vo2max", type=float, required=True, help="VO2max ml/kg/min")
    parser.add_argument("--resting_hr", type=float, required=True, help="안정시 심박수 bpm")
    parser.add_argument("--height", type=float, required=True, help="키 cm")
    parser.add_argument("--age", type=float, required=True, help="나이")
    parser.add_argument("--sex", type=str, required=True, help="성별(남=1/여=0 또는 남/여/M/F)")
    parser.add_argument("--no_plot", action="store_true", help="그래프 표시 안 함")
    args = parser.parse_args()

    # 1) Polar 로드/리샘플
    dfp_num = load_polar(args.polar)
    # 2) Analyzer 로드/리샘플
    dfa = load_analyzer(args.analyzer, dfp_num.index.min())

    # 3) τ 추정 및 적용
    best_tau = estimate_best_lag(dfp_num, dfa, LAG_SEARCH_RANGE)
    dfa_shift = dfa.copy()
    dfa_shift.index = dfa_shift.index + pd.Timedelta(seconds=best_tau)

    # 4) 시간 병합
    merged = pd.merge_asof(
        dfp_num.sort_index(),
        dfa_shift.sort_index(),
        left_index=True, right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=1)
    ).dropna()

    if merged.empty:
        raise RuntimeError("병합 결과가 비었습니다. 시각 범위/지연 설정/CSV 확인 필요.")

    # 5) 시각화 (선택)
    if not args.no_plot:
        fig, ax1 = plt.subplots(figsize=(12,6))
        # if "hr" in merged.columns:
        #     ax1.plot(merged.index, merged["hr"], label="Polar HR (bpm)")
        # elif "avgHr" in merged.columns:
        #     ax1.plot(merged.index, merged["avgHr"], label="Polar avgHR (bpm)")
        # ax1.set_ylabel("Heart Rate (bpm)")
        # ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(merged.index, merged["vo2_polar_mlmin"], label="Polar VO₂ (raw, ml/min)", alpha=0.7)
        ax2.plot(merged.index, merged["vo2_true_mlmin"], label="Analyzer VO₂ (ml/min)", linewidth=2)
        ax2.set_ylabel("VO₂ (ml/min)")

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax2.legend(l1 + l2, lb1 + lb2, loc="upper left")

        plt.title("Polar HR & Polar VO₂ vs Analyzer VO₂ (1s resampled, τ applied)")
        plt.tight_layout()
        plt.show()

    # 6) 학습(보정) — 사용자 특성 포함
    user_info = {
        "weight": args.weight,
        "vo2max": args.vo2max,
        "resting_hr": args.resting_hr,
        "height": args.height,
        "age": args.age,
        "sex": args.sex,
    }
    result = run_training(merged, user_info)

    # 7) 저장
    os.makedirs("out", exist_ok=True)

    # 보정 모델(계수 + τ + 피처 목록) 저장 — 앱에서 그대로 사용 가능
    best_model = {
        "type": result["best_name"],
        "features": result["feat_cols"],
        "coef": result["hub"][0],          # Huber 계수
        "intercept": result["hub"][1],     # Huber 절편
        "tau_sec": int(best_tau)
    }
    with open(os.path.join("out", "calibration_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_model, f, ensure_ascii=False, indent=2)

    # 병합+예측 시계열 저장
    out_df = merged.copy()
    out_df["pred_vo2_mlmin"] = result["best_name"]  # 라벨로 잠깐 표기
    # Huber 예측 다시 계산해서 컬럼 추가
    X_best = merged[result["feat_cols"]].values
    hub_coef, hub_intercept = np.array(result["hub"][0]), result["hub"][1]
    out_df["pred_vo2_mlmin"] = X_best.dot(hub_coef) + hub_intercept
    out_df.to_csv(os.path.join("out", "merged_with_prediction.csv"), index=True)

    # 보정 전/후 시각화
    if not args.no_plot:
        plt.figure(figsize=(12,6))
        plt.plot(out_df.index, out_df["vo2_true_mlmin"], label="Analyzer VO₂ (true)", linewidth=2)
        plt.plot(out_df.index, out_df["vo2_polar_mlmin"], label="Polar VO₂ (raw)", alpha=0.6)
        plt.plot(out_df.index, out_df["pred_vo2_mlmin"], label=f"Polar VO₂ (calibrated: {result['best_name']})", linestyle="--", linewidth=2)
        plt.ylabel("VO₂ (ml/min)")
        plt.title("VO₂: Analyzer vs Polar(raw) vs Polar(calibrated)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\n[Saved]")
    print(" - out/calibration_model.json  (보정식 계수 + τ + 피처)")
    print(" - out/merged_with_prediction.csv  (실측/원본/보정 시계열)")

if __name__ == "__main__":
    main()
