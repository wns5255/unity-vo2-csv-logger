# -*- coding: utf-8 -*-

import os, re, glob, json, argparse, warnings, math, pickle, re as _re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)

TIME_CAND_TOKENS = ['time', 'timestamp', 'datetime', 'date']

def _session_feats(g: pd.DataFrame, hr_col: str='hr') -> Dict[str, float]:
    hr = pd.to_numeric(g.get(hr_col, np.nan), errors='coerce')
    dt_sec = g['dt'].diff().dt.total_seconds().clip(lower=0).fillna(0)
    dur_min = float(dt_sec.sum() / 60.0)

    def _nanperc(x, q):
        x = pd.to_numeric(x, errors='coerce')
        return float(np.nanpercentile(x, q)) if np.isfinite(x).any() else np.nan

    # 없으면 NaN 시리즈로
    slope = pd.to_numeric(g.get('hr_slope_bpmps', np.nan), errors='coerce')
    hrr   = pd.to_numeric(g.get('hrr_calc',        np.nan), errors='coerce')

    feats = {
        'sess_hr_med'    : float(np.nanmedian(hr)),
        'sess_hr_p90'    : _nanperc(hr, 90),
        'sess_hr_std'    : float(np.nanstd(hr)),
        'sess_dur_min'   : dur_min,
        'sess_slope_mean': float(np.nanmean(slope)),
        'sess_hrr_mean'  : float(np.nanmean(hrr)),
    }
    hod = g['dt'].dt.hour + g['dt'].dt.minute/60.0
    feats['tod_sin'] = float(np.nanmean(np.sin(2*np.pi*hod/24)))
    feats['tod_cos'] = float(np.nanmean(np.cos(2*np.pi*hod/24)))
    return feats

def iso_predict_with_linear_tails(iso, x):
    x = np.asarray(x, float)
    xt, yt = iso.X_thresholds_, iso.y_thresholds_
    y = np.interp(np.clip(x, xt[0], xt[-1]), xt, yt)
    # 왼쪽/오른쪽 기울기
    ls = (yt[1] - yt[0]) / max(1e-12, (xt[1] - xt[0]))
    rs = (yt[-1] - yt[-2]) / max(1e-12, (xt[-1] - xt[-2]))
    left  = x < xt[0]
    right = x > xt[-1]
    y[left]  = yt[0]    + ls * (x[left]  - xt[0])
    y[right] = yt[-1]   + rs * (x[right] - xt[-1])
    return y


def _to_datetime_any(x: pd.Series) -> pd.Series:
    # 0) 이미 datetime → tz 제거
    if np.issubdtype(x.dtype, np.datetime64):
        dt = pd.to_datetime(x, errors='coerce', utc=True)
        return dt.dt.tz_convert(None)

    # 1) 숫자(epoch 계열) 먼저 시도 (문자열이어도 전부 숫자면 처리)
    x_num0 = pd.to_numeric(x, errors='coerce')
    if x_num0.notna().mean() > 0.9:
        med = x_num0.dropna().median()
        if med > 1e15: unit = 'ns'
        elif med > 1e12: unit = 'us'
        elif med > 1e11: unit = 'ms'
        elif med > 1e9:  unit = 's'
        elif med > 1e5:  unit = 's'
        else:
            base = pd.Timestamp('1970-01-01')
            return (pd.to_timedelta(x_num0, unit='s') + base)
        dt = pd.to_datetime(x_num0, unit=unit, errors='coerce', utc=True)
        return dt.dt.tz_convert(None)

    x_str = x.astype(str).str.strip()
    x_str = x_str.replace({'': np.nan})

    # 2) 전부 숫자인 문자열(10~16자리) → epoch로 시도
    digits = x_str.str.match(r'^\d{10,16}$')
    if digits.mean() > 0.5:
        x_num = pd.to_numeric(x_str.where(digits, np.nan), errors='coerce')
        med = x_num.dropna().median()
        if med > 1e15: unit = 'ns'
        elif med > 1e12: unit = 'us'
        elif med > 1e11: unit = 'ms'
        else:           unit = 's'
        dt = pd.to_datetime(x_num, unit=unit, errors='coerce', utc=True)
        if dt.notna().mean() > 0.5:
            return dt.dt.tz_convert(None)

    # 3) ISO 유사 문자열 정규화
    iso = x_str.str.replace('z', 'Z', regex=False)
    # Z → +00:00
    iso = iso.str.replace('Z', '+00:00', regex=False)
    # +0900 → +09:00
    iso = iso.str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
    # 소수 7자리 이상 → 6자리로 컷(파싱 실패 방지)
    iso = iso.str.replace(r'(\.\d{6})\d+', r'\1', regex=True)
    # 공백 뒤 'UTC' 같은 꼬리표 제거
    iso = iso.str.replace(r'\s+UTC$', '', regex=True)

    dt = pd.to_datetime(iso, errors='coerce', utc=True)
    if dt.notna().mean() > 0.5:
        return dt.dt.tz_convert(None)

    # 4) 일반 파서 (타임존/소수점 유무 가리지 않음)
    dt = pd.to_datetime(x_str, errors='coerce', utc=True)
    if dt.notna().mean() > 0.5:
        return dt.dt.tz_convert(None)

    # 5) HH:MM[:SS[.fff]] 형태(날짜 없음)
    time_like_mask = x_str.str.match(r'^\d{1,2}:\d{1,2}(:\d{1,2}(\.\d{1,9})?)?$')
    if time_like_mask.mean() > 0.5:
        base = pd.Timestamp('1970-01-01')
        td = pd.to_timedelta(x_str.where(time_like_mask, np.nan), errors='coerce')
        return (base + td)

    # 실패 시 NaT
    return pd.Series(pd.NaT, index=x.index, dtype='datetime64[ns]')

def find_time_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cand = [c for c in df.columns if any(tok in c.lower() for tok in TIME_CAND_TOKENS)]
    best_col, best_ratio = None, 0.0
    for col in cand:
        dt = _to_datetime_any(df[col])
        ratio = dt.notna().mean()
        if ratio > best_ratio:
            best_ratio, best_col = ratio, col
    return best_col

def coerce_datetime(df: pd.DataFrame, prefer_col: Optional[str]=None) -> Tuple[pd.DataFrame, str]:
    if prefer_col and prefer_col in df.columns:
        tcol = prefer_col
    else:
        tcol = find_time_column(df)
    if not tcol:
        raise ValueError("시간 컬럼을 찾지 못했습니다. --polar-time-col 또는 --real-time-col을 명시하세요.")

    dt = _to_datetime_any(df[tcol])
    ok_ratio = dt.notna().mean()

    # ★ 명시 지정이면 문턱 완화(≥1%만 있어도 진행), 자동 탐지면 기존 50% 유지
    min_ratio = 0.01 if (prefer_col is not None) else 0.5
    if ok_ratio < min_ratio:
        raise ValueError(f"시간 컬럼 '{tcol}'을 datetime으로 변환하지 못했습니다.")

    out = df.loc[dt.notna()].copy()
    # tz 제거
    dd = pd.to_datetime(dt.loc[dt.notna()], errors='coerce', utc=True)
    out['dt'] = dd.dt.tz_convert(None)

    out = out.sort_values('dt').reset_index(drop=True)
    return out, 'dt'


def pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        for c in df.columns:
            if name.lower() == c.lower():
                return c
        cand = [lower_map[k] for k in lower_map.keys() if name.lower() in k]
        if cand:
            cand.sort(key=len)
            return cand[0]
    return None

def detect_hr_col(df: pd.DataFrame) -> Optional[str]:
    return pick_col(df, ['HR', 'heart_rate', 'bpm', 'hr'])

def detect_polar_vo2_col(df: pd.DataFrame) -> Optional[str]:
    c = pick_col(df, ['vo2Abs_Lmin', 'VO2_Lmin', 'vo2_lmin', 'vo2(l/min)', 'vo2 (l/min)'])
    if c: return c
    c = pick_col(df, ['VO2', 'VO2_ml_min', 'VO2 (mL/min)'])
    return c

def detect_real_vo2_col(df: pd.DataFrame) -> Optional[str]:
    c = pick_col(df, ['VO2', 'VO2 (mL/min)', 'VO2_ml_min', 'VO2_mLmin'])
    if c: return c
    c = pick_col(df, ['VO2_Lmin', 'VO2 (L/min)', 'vo2_lmin'])
    return c

def detect_polar_phase_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    pr = [c for c in cols if 'phase' in c.lower() and c.lower().endswith('_polar')]
    if pr:
        return pr[0]
    pr = [c for c in cols if 'phase' in c.lower()]
    return pr[0] if pr else None

def detect_polar_active_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    cand1 = [c for c in cols if c.lower().endswith('_polar') and 'active' in c.lower()]
    if cand1:
        return cand1[0]
    cand2 = [c for c in cols if 'active' in c.lower()]
    if cand2:
        prefer = [c for c in cand2 if c.lower().endswith('_polar')]
        if prefer:
            return prefer[0]
        return cand2[0]
    cand3 = [c for c in cols if any(k in c.lower() for k in ['isactive','is_active','activityflag'])]
    return cand3[0] if cand3 else None

def lead_seconds(dt: pd.Series, seconds: float) -> pd.Series:
    return dt - pd.to_timedelta(seconds, unit='s')

def merge_time_nearest(polar: pd.DataFrame, real: pd.DataFrame,
                       tol_sec: float = 0.5) -> pd.DataFrame:
    tol = pd.Timedelta(seconds=tol_sec)
    merged = pd.merge_asof(
        polar.sort_values('dt'),
        real.sort_values('dt'),
        on='dt', direction='nearest', tolerance=tol,
        suffixes=('_POLAR','_REAL')
    )
    return merged

def to_lmin(series: pd.Series, assume: str = 'auto') -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if assume == 'l':
        return s
    if assume == 'ml':
        return s / 1000.0
    med = s.dropna().median() if s.notna().any() else np.nan
    if pd.isna(med):
        return s
    if 100 < med < 8000:
        return s / 1000.0
    return s

def ema(series: pd.Series, span_sec: Optional[float], freq_hz: float) -> pd.Series:
    if not span_sec or span_sec <= 0:
        return pd.to_numeric(series, errors='coerce')
    alpha = 1.0 - math.exp(-1.0 / (span_sec * freq_hz))
    return pd.to_numeric(series, errors='coerce').ewm(alpha=alpha, adjust=False).mean()

def parse_lag_spec(spec: str) -> List[int]:
    m = _re.match(r'^\s*(-?\d+)\s*:\s*(-?\d+)\s*:\s*(\d+)\s*$', spec or "")
    if not m:
        return list(range(-30, 31, 1))
    s, e, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if st <= 0: st = 1
    if e < s: s, e = e, s
    return list(range(s, e, st))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])) * 100.0

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def _safe_r2(y, p):
    try:
        if len(y) > 2 and np.all(np.isfinite(y)) and np.all(np.isfinite(p)):
            return float(r2_score(y, p))
    except Exception:
        pass
    return float('nan')

def _looks_like_real_vo2_col(col: str, real_target: str) -> bool:
    c = col.lower()
    tgt = (real_target or '').lower()
    if tgt and tgt in c:
        return True
    if ('real' in c and 'vo2' in c):
        return True
    if 'vo2_real' in c or 'real_vo2' in c:
        return True
    if ('vo2' in c) and any(u in c for u in ['lmin','mlmin','(l/min)','(ml/min)','_lmin','_mlmin']):
        if 'real' in c:
            return True
    return False

def _erode_active_mask_by_time(mask: np.ndarray, t: pd.Series, border_sec: float) -> np.ndarray:
    """시간축 기준으로 active 구간 앞/뒤를 border_sec만큼 제외"""
    if mask is None or border_sec <= 0 or mask.sum() == 0 or t is None or len(t)==0:
        return mask
    m = mask.astype(bool).copy()
    t = pd.to_datetime(t)
    N = len(m)
    bs = pd.to_timedelta(border_sec, unit='s')
    in_run = False
    start_idx = None
    for i in range(N+1):
        cur = (m[i] if i < N else False)
        prev = (m[i-1] if i > 0 else False)
        if cur and not prev:
            start_idx = i; in_run = True
        if (not cur) and prev and in_run:
            end_idx = i-1
            t_start = t.iloc[start_idx]; t_end = t.iloc[end_idx]
            left = t_start + bs; right = t_end - bs
            if right <= left:
                m[start_idx:end_idx+1] = False
            else:
                for j in range(start_idx, end_idx+1):
                    tj = t.iloc[j]
                    if (tj < left) or (tj > right):
                        m[j] = False
            in_run = False; start_idx = None
    return m

def train_model(merged: pd.DataFrame, args) -> Dict:
    df = merged.copy()

    # === NEW: 범주형 원-핫 인코딩 ===
    cat_cols = [c for c in (getattr(args, 'cat_cols', []) or []) if c in df.columns]
    dummy_cols = []
    if cat_cols:
        dummies = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols, prefix_sep='=')
        dummy_cols = list(dummies.columns)
        df = pd.concat([df, dummies], axis=1)
    # ================================
    y = pd.to_numeric(df[args.real_vo2_col], errors='coerce')
    if args.real_unit.lower() == 'ml':
        y = y / 1000.0
    elif args.real_unit.lower() == 'auto':
        y = to_lmin(y, 'auto')
    base = None
    baseline_name = None
    if args.use_residual and args.polar_vo2_col and args.polar_vo2_col in df.columns:
        base_raw = pd.to_numeric(df[args.polar_vo2_col], errors='coerce')
        base = to_lmin(base_raw, 'auto')
        baseline_name = args.polar_vo2_col
    drop_cols = set(['dt', args.real_vo2_col])
    if args.polar_vo2_col:
        drop_cols.add(args.polar_vo2_col)
    cand_num, leakage_cols = [], []
    for c in df.columns:
        if c in drop_cols:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        # ★ Inf/-Inf는 NaN으로 치환
        s.replace([np.inf, -np.inf], np.nan, inplace=True)

        if s.notna().any():
            if _looks_like_real_vo2_col(c, args.real_vo2_col):
                leakage_cols.append(c); continue
            df[c] = s
            cand_num.append(c)
    if not cand_num:
        raise ValueError("학습 가능한 숫자 피처가 없습니다. 입력 CSV의 컬럼을 확인하세요.")
    X = df[cand_num].copy()
    non_allnan_cols = [c for c in X.columns if X[c].notna().any()]
    dropped_allnan = sorted(list(set(X.columns) - set(non_allnan_cols)))
    if dropped_allnan:
        print(f"[WARN] 전부 NaN인 컬럼 제거: {dropped_allnan}")
    X = X[non_allnan_cols]
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    col_std = X_imp.std(axis=0)
    keep_mask = col_std > 0
    if not keep_mask.any():
        raise ValueError("모든 피처가 상수입니다. 학습을 진행할 수 없습니다.")
    X_imp = X_imp[:, keep_mask]
    feature_names = [non_allnan_cols[i] for i, k in enumerate(keep_mask) if k]
    dropped_const = [non_allnan_cols[i] for i, k in enumerate(keep_mask) if not k]
    if dropped_const:
        print(f"[WARN] 상수 컬럼 제거: {dropped_const}")
    y_arr = np.asarray(y, dtype=float)
    valid_rows = np.isfinite(y_arr)
    if args.use_residual and base is not None:
        base_arr = np.asarray(base, dtype=float)
        valid_rows &= np.isfinite(base_arr)
    valid_rows &= np.isfinite(X_imp).all(axis=1)
    # --- active-only 학습(선택) ---
    if int(args.active_only) == 1:
        m_active, _ = compute_active_mask(df, args)
        if m_active is not None:
            m_active = (m_active.astype(bool))
            if len(m_active) == len(valid_rows):
                valid_rows = valid_rows & m_active
            else:
                print('[WARN] active-only 적용 생략: 마스크 길이 불일치')
    X_clean = X_imp[valid_rows]
    y_clean = y_arr[valid_rows]
    base_clean = None if base is None else np.asarray(base, dtype=float)[valid_rows]
    if X_clean.shape[0] < 10:
        raise ValueError(f"유효 학습 표본 수가 부족합니다 (N={X_clean.shape[0]}). merge tolerance를 늘리거나 입력 결측치를 점검하세요.")
    

    base_affine = None
    if args.use_residual and base is not None:
        # ACTIVE에서만 a,b(affine) 계산
        m_active, _ = compute_active_mask(df, args)
        m_valid = np.isfinite(y_arr) & np.isfinite(np.asarray(base, float))
        if m_active is not None and len(m_active) == len(m_valid):
            m_valid = m_valid & m_active.astype(bool)
        m_valid = m_valid & valid_rows
        if m_valid.sum() >= 10:
            by = y_arr[m_valid]; bb = np.asarray(base, float)[m_valid]
            if args.active_calibrate in ('meanvar', 'affine'):
                if args.active_calibrate == 'meanvar':
                    stdb, stdy = float(np.std(bb)), float(np.std(by))
                    if stdb >= 1e-8:
                        a0 = stdy / stdb
                        b0 = float(np.mean(by) - a0 * np.mean(bb))
                    else:
                        a0, b0 = 1.0, float(np.mean(by - bb))
                else:  # affine
                    a0, b0 = np.polyfit(bb, by, 1)
            else:
                a0, b0 = 1.0, 0.0

            base_affine = {'a': float(a0), 'b': float(b0)}
            # ★ 타깃을 "REAL - (a·BASE + b)" 로 치환
            y_tgt = y_clean - (a0 * base_clean + b0)
        else:
            y_tgt = (y_clean - base_clean)  # 기존 residual
    else:
        y_tgt = y_clean

    scaler = StandardScaler(); Xs = scaler.fit_transform(X_clean)
    model = Ridge(alpha=args.alpha, random_state=42) if args.model.lower()=='ridge' else HuberRegressor(alpha=args.alpha, epsilon=1.35)

    # 1) 학습
    model.fit(Xs, y_tgt)

    # 2) pre-ISO 예측
    resid_pred = model.predict(Xs)
    if args.use_residual and base_clean is not None:
        if base_affine:   # ★ 새 방식
            pred_raw = (base_affine['a'] * base_clean + base_affine['b']) + resid_pred
        else:             # 구 방식
            pred_raw = base_clean + resid_pred
    else:
        pred_raw = resid_pred

    # 3) Isotonic (clip → 학습엔 그대로 둠; 예측 시엔 tail-linear 사용)
    iso = None
    pred = pred_raw
    if args.use_isotonic:
        iso = IsotonicRegression(out_of_bounds='clip')
        pred = iso.fit_transform(pred_raw, y_clean)

    # 4) 메트릭/아티팩트
    metrics = {
        'N': int(len(y_clean)),
        'R2': _safe_r2(y_clean, pred),
        'MAE_Lmin': float(mean_absolute_error(y_clean, pred)),
        'RMSE_Lmin': float(rmse(y_clean, pred)),
        'MAPE_%': float(mape(y_clean, pred)),
        'Dropped_AllNaN_Cols': dropped_allnan,
        'Dropped_Const_Cols': dropped_const,
        'Dropped_Leakage_Cols': leakage_cols,
        'Used_Features': feature_names,
    }
    # (선택) ISO 학습 구간 범위 참고용으로 metrics에 저장
    if iso is not None:
        metrics['ISO_X_min'] = float(iso.X_thresholds_[0])
        metrics['ISO_X_max'] = float(iso.X_thresholds_[-1])

    artifact = {
        'imputer': imputer, 'scaler': scaler, 'model': model,
        'use_residual': bool(args.use_residual and (baseline_name is not None)),
        'baseline_col': baseline_name, 'iso': iso,
        'feature_names': feature_names, 'real_vo2_col': args.real_vo2_col,
        'real_unit': args.real_unit, 'metrics': metrics, 'importances': {},
        'cat_cols': cat_cols, 'dummy_cols': dummy_cols,
        'baseline_affine': base_affine,  # ★ 추가

    }
    merged_clean = merged.loc[valid_rows].copy().reset_index(drop=True)
    return artifact, pred, y_clean, merged_clean

def _find_src_columns(cols):
    cols = list(cols)
    polar_candidates = [c for c in cols if _re.match(r'^__src__.*POLAR$', c)]
    real_candidates  = [c for c in cols if _re.match(r'^__src__.*REAL$', c)]
    if not polar_candidates:
        polar_candidates = [c for c in cols if _re.match(r'^__src__.*_x$', c)]
    if not real_candidates:
        real_candidates  = [c for c in cols if _re.match(r'^__src__.*_y$', c)]
    single = '__src__' if '__src__' in cols else None
    polar_src = polar_candidates[0] if polar_candidates else (single if single else None)
    real_src  = real_candidates[0]  if real_candidates  else None
    return polar_src, real_src

def _sanitize(s: str) -> str:
    s = str(s)
    s = _re.sub(r'[\\/:*?"<>|]+', '_', s)
    s = _re.sub(r'\s+', '_', s).strip('_')
    if len(s) > 150:
        s = s[:150]
    return s or 'NA'

def detect_polar_phase_or_active_col(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    ph = detect_polar_phase_col(df)
    ac = detect_polar_active_col(df)
    return ph, ac

def _compute_active_mask_from_phase(df: pd.DataFrame, phase_col: str, active_words: List[str]) -> np.ndarray:
    s = df[phase_col].astype(str).str.strip().str.lower()
    words = set([w.strip().lower() for w in active_words if w.strip()])
    return s.isin(words).to_numpy()

def _compute_active_mask_polar(df: pd.DataFrame, args) -> Tuple[Optional[np.ndarray], Optional[str]]:
    phase_col, active_col = detect_polar_phase_or_active_col(df)
    if phase_col and phase_col in df.columns:
        mask = _compute_active_mask_from_phase(df, phase_col, args.phase_active_values)
        return mask, phase_col
    if active_col and active_col in df.columns:
        s = df[active_col]
        if s.dtype == bool:
            mask = s.values
        else:
            s2 = s.astype(str).str.strip().str.lower()
            mask = s2.isin(['1','true','t','y','yes','on','active']).to_numpy()
            try:
                num = pd.to_numeric(s, errors='coerce')
                mask = np.where(np.isfinite(num), num > 0, mask)
            except Exception:
                pass
        return mask.astype(bool), active_col
    return None, None

def _compute_active_mask_hr(df: pd.DataFrame, args) -> Tuple[Optional[np.ndarray], Optional[str]]:
    hr_col = detect_hr_col(df)
    if not hr_col or hr_col not in df.columns:
        return None, None
    hr = pd.to_numeric(df[hr_col], errors='coerce')
    if hr.notna().sum() < 5:
        return None, hr_col
    rest_hr = args.rest_hr if args.rest_hr > 0 else (np.nanmedian(hr) - 10)
    mask = (hr >= (rest_hr + args.active_hr_delta)).fillna(False).to_numpy()
    return mask, hr_col

def compute_active_mask(df: pd.DataFrame, args) -> Tuple[Optional[np.ndarray], str]:
    src = (args.active_mask_from or 'polar').lower()
    if src == 'polar':
        m, c = _compute_active_mask_polar(df, args)
        if m is None:
            warnings.warn("[WARN] Polar phase/active 컬럼을 찾지 못했습니다. (--active-mask-from polar)")
        # ★ 가장자리 제외
        m = _erode_active_mask_by_time(m, df['dt'], args.active_border_sec) if m is not None else None
        return m, (c or 'phase/active(POLAR)')
    elif src == 'hr':
        m, c = _compute_active_mask_hr(df, args)
        m = _erode_active_mask_by_time(m, df['dt'], args.active_border_sec) if m is not None else None
        return m, (c or 'HR(threshold)')
    else:  # auto
        m, c = _compute_active_mask_polar(df, args)
        if m is not None:
            m = _erode_active_mask_by_time(m, df['dt'], args.active_border_sec)
            return m, c or 'phase/active(POLAR)'
        m, c = _compute_active_mask_hr(df, args)
        m = _erode_active_mask_by_time(m, df['dt'], args.active_border_sec) if m is not None else None
        return m, (c or 'HR(threshold)')

def _shade_active(ax, t: pd.Series, active_mask: np.ndarray, alpha: float = 0.12):
    if active_mask is None or t is None or len(t) == 0:
        return
    t = pd.to_datetime(t)
    in_span = False; start = None
    for i, flag in enumerate(active_mask):
        if bool(flag) and not in_span:
            in_span = True; start = t.iloc[i]
        elif (not bool(flag)) and in_span:
            end = t.iloc[i]; ax.axvspan(start, end, color='grey', alpha=alpha, zorder=0); in_span=False
    if in_span:
        ax.axvspan(start, t.iloc[-1], color='grey', alpha=alpha, zorder=0)

def _mean_ape_pct(y_mean, p_mean) -> float:
    if y_mean is None or not np.isfinite(y_mean) or y_mean == 0:
        return float('nan')
    return float(abs(p_mean - y_mean) / y_mean * 100.0)

# 기존 calibrate_predictions 함수를 이 코드로 대체하세요.

def calibrate_predictions(merged_clean, y, pred, args):
    df = merged_clean.copy()
    df['__Y__'] = y
    df['__PRED__'] = pred
    
    # 결과 담을 배열 복사
    pred_cal_s = pd.Series(pred, index=df.index, dtype=float).copy()
    params = {}

    # ---------------------------------------------------------
    # [NEW] 참조 테이블(Ref Table) 기반 강제 보정 로직
    # ---------------------------------------------------------
    if getattr(args, 'ref_table', None) and os.path.exists(args.ref_table):
        print(f"[INFO] Reference Table 보정 시작: {args.ref_table}")
        try:
            # CSV 읽기 (subject, motion_id, target_vo2_ml 컬럼이 있어야 함)
            ref_df = pd.read_csv(args.ref_table, encoding='utf-8')
            # 공백 제거 및 문자열 처리
            ref_df['subject'] = ref_df['subject'].astype(str).str.strip()
            ref_df['motion_id'] = ref_df['motion_id'].astype(str).str.strip()
            
            # 검색 속도를 위해 Dictionary로 변환 {(subject, motion): target_val}
            ref_map = {}
            for _, row in ref_df.iterrows():
                key = (row['subject'], row['motion_id'])
                ref_map[key] = float(row['target_vo2_ml'])

            # 데이터프레임에 subject, motion_id가 있는지 확인
            if 'subject' in df.columns and 'motion_id' in df.columns:
                # 그룹별로 순회 (파일명 기준이 아니라 사용자/동작 기준으로 그룹핑)
                # 원본 파일명(__src__) 단위로 루프를 돌면서 해당 파일의 subject/motion을 찾습니다.
                polar_src, _ = _find_src_columns(df.columns)
                grp_col = polar_src if polar_src else '__src__'
                
                for src_key, g in df.groupby(grp_col):
                    idx = g.index
                    # 해당 세션의 subject, motion_id 추출 (모든 행이 같다면 첫번째 값 사용)
                    # 데이터에 subject/motion_id 컬럼이 문자열로 들어있어야 합니다.
                    subj = str(g['subject'].iloc[0]).strip()
                    mot  = str(g['motion_id'].iloc[0]).strip()
                    
                    target_ml = ref_map.get((subj, mot))
                    
                    if target_ml is not None:
                        # 타겟값(ml)을 L/min으로 변환 (예: 470 -> 0.470)
                        target_lmin = target_ml / 1000.0
                        
                        # 현재 예측값의 평균 계산 (Active 구간만 할지, 전체 할지 결정)
                        # 여기서는 args.calibrate_active_only 옵션을 따름
                        mask_for_calc = None
                        if int(getattr(args, "calibrate_active_only", 1)) == 1:
                            m_act, _ = compute_active_mask(g, args)
                            if m_act is not None and m_act.sum() > 5:
                                mask_for_calc = m_act.astype(bool)
                        
                        # 마스크가 없으면 전체 구간 평균
                        current_vals = pred_cal_s.loc[idx].values
                        if mask_for_calc is not None:
                            current_mean = np.nanmean(current_vals[mask_for_calc])
                        else:
                            current_mean = np.nanmean(current_vals)
                            
                        # 보정량 계산 (Bias 방식: 목표 - 현재평균)
                        offset = target_lmin - current_mean
                        
                        # 적용: 해당 세션 전체 값을 이동시킴
                        pred_cal_s.loc[idx] = pred_cal_s.loc[idx] + offset
                        
                        params[str(src_key)] = {
                            'type': 'ref_table_bias', 
                            'subject': subj, 
                            'motion': mot,
                            'target_ml': target_ml,
                            'offset_applied': offset
                        }
                        # print(f"   -> {subj}/{mot} : Target={target_lmin:.3f}, Cur={current_mean:.3f}, Off={offset:+.3f}")
                    else:
                        # 테이블에 없는 경우 기존 로직(args.active_calibrate)을 따르거나 패스
                        pass
                        
            else:
                print("[WARN] 데이터에 'subject' 또는 'motion_id' 컬럼이 없어 Ref Table 보정을 건너뜁니다.")

        except Exception as e:
            print(f"[ERROR] Ref Table 보정 중 오류 발생: {e}")
    
    # Ref Table 보정이 수행되었다면 기존 calibrate 로직(meanvar/affine 등)은 
    # 중복 적용을 막기 위해 건너뛰거나, Ref Table에 없는 데이터에만 적용해야 합니다.
    # 여기서는 Ref Table이 적용되지 않은(params에 키가 없는) 데이터에 대해서만 기존 로직 수행
    
    polar_src, real_src = _find_src_columns(df.columns)
    grp_cols = [c for c in [polar_src, real_src] if c]
    
    if args.active_calibrate != 'none' and grp_cols:
        # 기존 로직 (Ref Table 보정 안 된 것만)
        for keys, g in df.groupby(grp_cols, dropna=False):
            # 이미 보정된 키라면 스킵
            src_key = str(keys[0]) if isinstance(keys, tuple) else str(keys)
            if src_key in params: 
                continue 
            
            # ... (이하 기존 로직과 동일) ...
            idx = g.index
            m, _ = compute_active_mask(g, args)
            if m is None or m.sum() < 5: continue
            yv = g.loc[m, '__Y__'].values
            pv = g.loc[m, '__PRED__'].values
            
            # (기존 meanvar, bias, affine 로직 복사 붙여넣기)
            if args.active_calibrate == 'meanvar':
                stdb, stdy = float(np.std(pv)), float(np.std(yv))
                if stdb >= 1e-8:
                    a = stdy / stdb
                    b = float(np.mean(yv) - a * np.mean(pv))
                    pred_cal_s.loc[idx] = a * pred_cal_s.loc[idx] + b
                    params[str(keys)] = {'type':'meanvar', 'a':a, 'b':b}
                else:
                    offset = float(np.mean(pv) - np.mean(yv))
                    pred_cal_s.loc[idx] = pred_cal_s.loc[idx] - offset
                    params[str(keys)] = {'type':'bias', 'offset':offset}
            
            elif args.active_calibrate == 'bias':
                offset = float(np.mean(pv) - np.mean(yv))
                pred_cal_s.loc[idx] = pred_cal_s.loc[idx] - offset
                params[str(keys)] = {'type':'bias','offset':offset}

            elif args.active_calibrate == 'affine':
                a,b = np.polyfit(pv, yv, 1)
                pred_cal_s.loc[idx] = a*pred_cal_s.loc[idx] + b
                params[str(keys)] = {'type':'affine','a':float(a),'b':float(b)}

    return pred_cal_s.to_numpy(), params

def predict_from_polar_only(args):
    # 1) 아티팩트 로드
    did_apply_calib = False   # ★ 추가: 실제 캘리브레이션 적용 여부
    art_path = args.load_artifact or str(Path(args.out) / "model_artifact.pkl")
    with open(art_path, "rb") as f:
        art = pickle.load(f)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 새 Polar 데이터 적재 + 시간 정제
    polar = load_csvs(args.predict_from_polar)
    polar, _ = coerce_datetime(polar, prefer_col=args.polar_time_col)
    polar = add_hr_features(polar, dt_col='dt', hr_col=args.hr_col or 'hr', resting_col='restingHr')


    # === NEW: 학습 때의 범주형 더미 복원 ===
    cat_cols   = art.get('cat_cols', []) or []
    dummy_cols = art.get('dummy_cols', []) or []
    if cat_cols:
        dnew = pd.get_dummies(polar[cat_cols].astype(str), prefix=cat_cols, prefix_sep='=')
        # 학습에서 본 dummy들만 정확히 채워 넣기 (없으면 0)
        for c in dummy_cols:
            if c in dnew.columns:
                polar[c] = dnew[c].astype(int)
            else:
                polar[c] = 0
    # ======================================

    # (선택) EMA 적용
    if args.ema_sec > 0:
        num_cols = [c for c in polar.columns if np.issubdtype(polar[c].dtype, np.number)]
        # 샘플링 주기 추정
        if len(polar) > 1:
            med_dt = (polar['dt'].diff().dt.total_seconds().dropna().median() or 1.0)
        else:
            med_dt = 1.0
        freq_hz = 1.0 / max(med_dt, 1e-6)
        for c in num_cols:
            polar[c] = ema(polar[c], args.ema_sec, freq_hz)

    # 3) 학습 때 쓰인 피처 셋 재현 + Imputer가 기억하는 전체 피처 이름 정합 맞추기
    feat_names = art['feature_names']  # 학습에서 최종 사용(상수/올NaN 제거 후) 컬럼
    imputer = art['imputer']
    scaler  = art['scaler']
    model   = art['model']

    # Imputer가 fit될 때 보았던 피처 이름 (sklearn >=1.0)
    if hasattr(imputer, "feature_names_in_"):
        imputer_feats = list(imputer.feature_names_in_)
    else:
        # 구버전 대비: 정보가 없으면 feat_names로 가정
        imputer_feats = list(feat_names)

    # Imputer가 본 전체 피처셋으로 데이터프레임 구성(없는 컬럼은 NaN → 중앙값 대체)
    X_full = pd.DataFrame(index=polar.index)
    for c in imputer_feats:
        if c in polar.columns:
            X_full[c] = pd.to_numeric(polar[c], errors='coerce')
        else:
            X_full[c] = np.nan

    # X_full 만든 직후에 추가
    present = {c: (c in polar.columns) for c in imputer_feats}
    missing = [c for c, ok in present.items() if not ok]
    print("[DEBUG] features seen by imputer:", imputer_feats)
    print("[DEBUG] missing in predict file:", missing)

    # 가용 피처들의 변동성 체크

    def _std(s): 
        try: return float(np.nanstd(pd.to_numeric(s, errors='coerce')))
        except: return float('nan')
    dbg_cols = sorted(set(art['feature_names'] + ([art['baseline_col']] if art.get('use_residual') else [])))
    stats = {c: _std(polar[c]) if c in polar.columns else None for c in dbg_cols}
    print("[DEBUG] per-column std on predict data:", stats)

    # Imputer 변환 (전체 피처 기준)
    X_imp_full = imputer.transform(X_full[imputer_feats])

    # 학습 때 최종 사용한 feat_names 순서/부분집합만 인덱싱해서 스케일링/예측
    # (imputer_feats → feat_names 매핑)
    name_to_idx = {name: i for i, name in enumerate(imputer_feats)}
    missing_for_model = [c for c in feat_names if c not in name_to_idx]
    if missing_for_model:
        print(f"[WARN] 모델이 기대하는 피처 중 Imputer 기준에 없는 항목: {missing_for_model}")

    keep_idx = [name_to_idx[c] for c in feat_names if c in name_to_idx]
    X_imp = X_imp_full[:, keep_idx]

    # 4) 동일 전처리 → PRE-ISO 예측 (학습과 동일하게 baseline을 여기서 더함)
    Xs = scaler.transform(X_imp)
    resid_pred = model.predict(Xs)

    pred_preiso = resid_pred
    if art.get('use_residual') and art.get('baseline_col') and art['baseline_col'] in polar.columns:
        base = to_lmin(polar[art['baseline_col']], 'auto').to_numpy()
        a0, b0 = 1.0, 0.0
        if art.get('baseline_affine'):
            a0 = float(art['baseline_affine'].get('a', 1.0))
            b0 = float(art['baseline_affine'].get('b', 0.0))
            print(f"[DEBUG] baseline_affine a={a0:.4f}, b={b0:.4f}")
        pred_preiso = (a0 * base + b0) + resid_pred   # ★

    # 5) ISO 적용 (한 번만, tail-linear)
    if art.get('iso', None) is not None:
        pred_iso = iso_predict_with_linear_tails(art['iso'], pred_preiso)
        w = float(getattr(args, 'iso_blend_w', 0.5))  # 0~1
        pred = w*pred_iso + (1-w)*pred_preiso
    else:
        pred = pred_preiso
    
    # -------------------------------------------------------------------------
    # 6) (선택) calibration 적용
    # -------------------------------------------------------------------------
    
    # [NEW] Ref Table(CSV) 기반 우선 보정
    did_apply_ref_table = False
    if getattr(args, 'ref_table', None) and os.path.exists(args.ref_table):
        try:
            print(f"[INFO] Ref Table 로드 중: {args.ref_table}")
            ref_df = pd.read_csv(args.ref_table, encoding='utf-8')
            # 공백 제거
            ref_df['subject'] = ref_df['subject'].astype(str).str.strip()
            ref_df['motion_id'] = ref_df['motion_id'].astype(str).str.strip()
            
            # 매핑용 딕셔너리 {(subject, motion): target_ml}
            ref_map = {}
            for _, r in ref_df.iterrows():
                ref_map[(r['subject'], r['motion_id'])] = float(r['target_vo2_ml'])

            # 현재 예측 데이터의 subject/motion 확인 (polar 데이터프레임 기준)
            # 만약 CSV 파일 안에 subject, motion_id 컬럼이 없다면 파일명 등에서 가져와야 하지만,
            # 현재는 파일 내부에 있다고 가정
            if 'subject' in polar.columns and 'motion_id' in polar.columns:
                # pandas Series로 변환해 수정 용이하게 함
                pred_s = pd.Series(pred, index=polar.index, dtype=float)
                
                # 원본 소스 파일별로 그룹핑하여 보정
                grp_col = '__src__' if '__src__' in polar.columns else None
                if grp_col:
                    for src_key, g in polar.groupby(grp_col):
                        idx = g.index
                        # 세션의 대표 subject/motion (첫번째 행 기준)
                        if len(g) > 0:
                            s_val = str(g['subject'].iloc[0]).strip()
                            m_val = str(g['motion_id'].iloc[0]).strip()
                            
                            target_ml = ref_map.get((s_val, m_val))
                            if target_ml is not None:
                                # [NEW] 랜덤 노이즈 추가: -3.0 ~ +3.0 사이의 실수
                                noise_ml = np.random.uniform(-3.0, 3.0)
                                final_target_ml = target_ml + noise_ml
                                
                                # ml -> L/min 변환
                                target_lmin = final_target_ml / 1000.0
                                                                
                                # 현재 세션의 평균 계산 (Active Only 옵션 고려)
                                m_act = None
                                if int(getattr(args, "calibrate_active_only", 1)) == 1:
                                    m_act, _ = compute_active_mask(g, args)
                                
                                # Active 구간이 충분하면 그 구간만 평균, 아니면 전체 평균
                                if m_act is not None and m_act.sum() > 5:
                                    cur_mean = np.nanmean(pred_s.loc[idx].values[m_act.astype(bool)])
                                else:
                                    cur_mean = np.nanmean(pred_s.loc[idx].values)
                                    
                                offset = target_lmin - cur_mean
                                
                                # 보정 적용 (전체 이동)
                                pred_s.loc[idx] = pred_s.loc[idx] + offset
                                did_apply_ref_table = True
                                did_apply_calib = True # bias-head 등 뒤쪽 로직에 영향 줌
                                
                                print(f"[INFO] Ref Table 보정 적용 ({os.path.basename(str(src_key))}): "
                                      f"{s_val}/{m_val} -> 목표 {target_lmin:.3f} (Noise {noise_ml:+.1f}ml), "
                                      f"현재 {cur_mean:.3f}, 보정 {offset:+.3f}")
                
                pred = pred_s.to_numpy()
            else:
                print("[WARN] Polar 파일에 subject/motion_id 컬럼이 없어 Ref Table을 적용할 수 없습니다.")
                
        except Exception as e:
            print(f"[ERROR] Ref Table 적용 중 오류: {e}")

    # [기존 로직] JSON 기반 보정 (Ref Table이 적용 안 된 경우에만 수행하거나, 중복 적용 정책 결정)
    # 여기서는 Ref Table이 적용되었다면 JSON 보정은 건너뛰도록 처리
    cal_params = None
    if not did_apply_ref_table:
        if getattr(args, "apply_calibration_from", None):
            try:
                with open(args.apply_calibration_from, "r", encoding="utf-8") as f:
                    cal_params = json.load(f)
            except Exception as e:
                print(f"[WARN] calibration params 로드 실패: {e}")

        if cal_params:
            pred_s = pd.Series(pred, index=polar.index, dtype=float)
            m_apply = None
            if int(getattr(args, "calibrate_active_only", 1)) == 1:
                m_apply, _ = compute_active_mask(polar, args)
                if m_apply is not None and len(m_apply) != len(pred_s):
                    m_apply = None

            # --- 여기 추가: 파일명 무시하고 GLOBAL만 적용 ---
            if int(getattr(args, "force_global_calib", 0)) == 1:
                if "__GLOBAL__" in cal_params:
                    chosen = cal_params["__GLOBAL__"]
                    typ = str(chosen.get("type", "bias")).lower()
                    idx = pred_s.index if m_apply is None else pred_s.index[np.array(m_apply, dtype=bool)]
                    if typ in ("meanvar","affine"):
                        a = float(chosen.get("a",1.0)); b = float(chosen.get("b",0.0))
                        pred_s.loc[idx] = a*pred_s.loc[idx] + b
                    else:
                        off = float(chosen.get("offset",0.0))
                        pred_s.loc[idx] = pred_s.loc[idx] - off
                    did_apply_calib = True
                    pred = pred_s.to_numpy()
                else:
                    print("[WARN] __GLOBAL__ calibration not found; skipping calibration")

            def first_polar_from_key(k: str) -> str:
                m = re.search(r"\('([^']+)'", k)
                return m.group(1) if m else None

            if "__src__" in polar.columns:
                for psrc, g in polar.groupby("__src__"):
                    psrc_base = os.path.basename(str(psrc))
                    chosen = None
                    for k, v in cal_params.items():
                        if first_polar_from_key(str(k)) == psrc_base:
                            chosen = v; break

                    # ★ Fallback 정책: 기본은 'none'
                    if chosen is None:
                        if getattr(args, "calib_fallback", "none") == "global" and '__GLOBAL__' in cal_params:
                            chosen = cal_params['__GLOBAL__']
                            # print(f"[WARN] no calibration for {psrc_base}; applying GLOBAL calibration")
                        else:
                            # print(f"[WARN] no calibration for {psrc_base}; skipping calibration fallback")
                            continue  # ← 이 세션은 캘리브레이션 미적용

                    idx = g.index
                    mask_idx = idx if m_apply is None else idx[np.array(m_apply, dtype=bool)[idx]]
                    typ = str(chosen.get("type", "bias")).lower()
                    if typ in ("meanvar","affine"):
                        a = float(chosen.get("a", 1.0)); b = float(chosen.get("b", 0.0))
                        pred_s.loc[mask_idx] = a*pred_s.loc[mask_idx] + b
                    else:
                        off = float(chosen.get("offset", 0.0))
                        pred_s.loc[mask_idx] = pred_s.loc[mask_idx] - off
                    did_apply_calib = True   # ★ 적용됨
            else:
                k0 = next(iter(cal_params.keys()))
                chosen = cal_params[k0]
                if getattr(args, "calib_fallback", "none") == "global" or k0 != "__GLOBAL__":
                    typ = str(chosen.get("type","bias")).lower()
                    if m_apply is None:
                        m_apply = np.ones(len(pred_s), dtype=bool)
                    if typ in ("meanvar","affine"):
                        a = float(chosen.get("a",1.0)); b = float(chosen.get("b",0.0))
                        pred_s.loc[m_apply] = a*pred_s.loc[m_apply] + b
                    else:
                        off = float(chosen.get("offset",0.0))
                        pred_s.loc[m_apply] = pred_s.loc[m_apply] - off
                    did_apply_calib = True   # ★ 적용됨
                else:
                    print("[WARN] skipping GLOBAL calibration (fallback=none)")
            pred = pred_s.to_numpy()

    # === HR-only 세션 오프셋 보정 (캘리브 적용 여부에 따라 clamp 다르게) ===
    use_bias  = int(getattr(args, "use_bias_head", 1)) == 1
    bias_head = art.get('bias_head', None)

    if use_bias and bias_head:
        pred_s = pd.Series(pred, index=polar.index, dtype=float)

        # shrink는 동일, clamp는 did_apply_calib 여부에 따라 선택
        shrink = float(getattr(args, "bias_head_shrink", 0.5))
        max_abs = float(getattr(args, "bias_head_max_abs_after_cal", 0.05)) if did_apply_calib \
                  else float(getattr(args, "bias_head_max_abs", 0.15))

        for psrc, g in polar.groupby("__src__"):
            feats = _session_feats(g, hr_col=args.hr_col or 'hr')
            df1   = pd.DataFrame([feats])[bias_head['feature_names']]
            X1    = bias_head['scaler'].transform(bias_head['imputer'].transform(df1))
            off_pred = float(bias_head['model'].predict(X1)[0])

            # 수축 + 클램프
            off_adj = float(np.clip(shrink * off_pred, -max_abs, max_abs))

            # (디버그) ACTIVE 평균 이동량 로깅
            m_sess, _ = compute_active_mask(g, args)
            if m_sess is not None and m_sess.any():
                pre_mean  = float(np.nanmean(pred_s.loc[g.index][m_sess]))
                post_mean = float(np.nanmean((pred_s.loc[g.index] + off_adj)[m_sess]))
                print(f"[DEBUG] bias-head mean shift (ACTIVE, {psrc}): "
                      f"{pre_mean:.3f} -> {post_mean:.3f} (Δ={post_mean-pre_mean:+.3f})")

            pred_s.loc[g.index] = pred_s.loc[g.index] + off_adj
            print(f"[INFO] bias-head offset applied to {psrc}: "
                  f"{off_pred:+.3f} -> {off_adj:+.3f} L/min (shrink={shrink}, clamp=±{max_abs}, "
                  f"{'after-cal' if did_apply_calib else 'no-cal'})")

        pred = pred_s.to_numpy()

    elif use_bias and bias_head and did_apply_calib:
        print("[WARN] calibration이 적용되어 bias-head는 건너뜁니다(이중보정 방지).")

        # === 세션별 스케일/오프셋 자동 보정 (affine-head) ===
    if int(getattr(args, "use_affine_head", 0)) == 1 and ('affine_head' in art):
        ah = art['affine_head']
        pred_s = pd.Series(pred, index=polar.index, dtype=float)
        for psrc, g in polar.groupby("__src__"):
            feats = _session_feats(g, hr_col=args.hr_col or 'hr')
            df1 = pd.DataFrame([feats])[ah['feature_names']]
            Xa  = ah['scaler'].transform(ah['imputer'].transform(df1))
            a_hat = float(ah['mdl_a'].predict(Xa)[0])
            b_hat = float(ah['mdl_b'].predict(Xa)[0])

            # 수축 + 클램프
            shrink_a = float(getattr(args, "affine_head_shrink_a", 0.5))
            shrink_b = float(getattr(args, "affine_head_shrink_b", 0.5))
            dev_max  = float(getattr(args, "affine_head_max_scale_dev", 0.25))
            b_max    = float(getattr(args, "affine_head_max_abs_b", 0.12))

            a_adj = 1.0 + np.clip(shrink_a * (a_hat - 1.0), -dev_max, dev_max)
            b_adj = np.clip(shrink_b * b_hat, -b_max, b_max)

            # 적용
            idx = g.index
            before_mean = float(np.nanmean(pred_s.loc[idx]))
            pred_s.loc[idx] = a_adj * pred_s.loc[idx] + b_adj
            after_mean = float(np.nanmean(pred_s.loc[idx]))
            print(f"[INFO] affine-head applied to {psrc}: a={a_hat:.3f}->{a_adj:.3f}, "
                  f"b={b_hat:+.3f}->{b_adj:+.3f}; mean {before_mean:.3f}->{after_mean:.3f}")
        pred = pred_s.to_numpy()


    # 7) ACTIVE/phase 계산 및 결과 저장 — pred_df는 **한 번만** 만든다
    m_active, _ = compute_active_mask(polar, args)
    phase_col, _ = detect_polar_phase_or_active_col(polar)
    phase_series = polar[phase_col] if (phase_col and phase_col in polar.columns) else None

    pred_df = polar[['dt']].copy()
    pred_df['time'] = pred_df['dt']
    pred_df['VO2_PREISO_Lmin'] = pred_preiso
    pred_df['VO2_PRED_Lmin']   = pred
    if art.get('use_residual') and art.get('baseline_col') in polar.columns:
        pred_df['POLAR_BASE_Lmin'] = to_lmin(polar[art['baseline_col']], 'auto').values

    if m_active is not None and len(m_active) == len(pred_df):
        pred_df['ACTIVE'] = pd.Series(m_active, index=pred_df.index).astype(int)
    else:
        pred_df['ACTIVE'] = np.nan
    if phase_series is not None:
        pred_df['phase'] = phase_series.astype(str).values

    if int(getattr(args, "plot_active_only", 1)) == 1 and m_active is not None:
        before_n = len(pred_df)
        pred_df = pred_df.loc[m_active.astype(bool)].copy()
        after_n = len(pred_df)
        print(f"[INFO] ACTIVE ONLY: kept {after_n} / {before_n} rows (border={args.active_border_sec}s)")

    pred_df.to_csv(out_dir / "pred_from_polar.csv", index=False, encoding="utf-8-sig")

    # pred_df 저장 직후에 추가
    plot_active_only = int(getattr(args, "plot_active_only", 1)) == 1
    # 8) 간단 타임시리즈 플롯 (액티브만)
    try:
        fig, ax = plt.subplots(figsize=(14,4.5))
        ax.plot(pred_df['dt'], pred_df['VO2_PRED_Lmin'], label='PRED VO₂ (L/min)', linewidth=1.8, alpha=0.9)
        if art.get('baseline_col') and art['baseline_col'] in polar.columns:
            base_series = to_lmin(polar.loc[pred_df.index, art['baseline_col']], 'auto')
            ax.plot(pred_df['dt'], base_series, label='POLAR VO₂ (baseline)', linewidth=1.2, alpha=0.7)
        # ACTIVE 음영(전체구간일 때만)
        if int(getattr(args, "shade_active", 1)) == 1 and (not plot_active_only) and (m_active is not None):
            _shade_active(ax, polar['dt'], m_active, alpha=float(getattr(args, 'shade_alpha', 0.12)))
        ax.set_xlabel('Time'); ax.set_ylabel('VO₂ (L/min)'); ax.set_title('VO₂ Prediction')
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "timeseries_vo2_predict.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[WARN] 예측 플롯 실패: {e}")

    print(f"[INFO] Saved: {out_dir / 'pred_from_polar.csv'}")


def add_hr_features(df: pd.DataFrame, dt_col='dt', hr_col='hr', resting_col='restingHr'):
    if hr_col not in df.columns or dt_col not in df.columns:
        return df
    df = df.copy()
    t = pd.to_datetime(df[dt_col])

    dt_raw = t.diff().dt.total_seconds()

    # 유효(>0)한 간격의 중앙값만 사용
    med_pos = float(dt_raw[(dt_raw > 0) & np.isfinite(dt_raw)].median() or 1.0)

    # NA는 중앙값으로, 0 또는 음수 간격은 NaN으로(기울기 계산 시 NaN → 안전)
    dt = dt_raw.fillna(med_pos)
    dt = dt.where(dt > 0, np.nan)

    hr = pd.to_numeric(df[hr_col], errors='coerce')

    # (1) 순간 기울기: dt==0/<=0인 곳은 NaN → Inf 방지
    df['hr_slope_bpmps'] = hr.diff().div(dt)

    # (2) 짧은 EMA/롤링 통계
    def _steps(sec): 
        return max(1, int(round(sec / (med_pos if med_pos > 0 else 1.0))))
    for w in (5, 15, 30):
        alpha = 1.0 / _steps(w)
        df[f'hr_ema_{w}s'] = hr.ewm(alpha=alpha, adjust=False).mean()
    for w in (10, 30):
        df[f'hr_std_{w}s'] = hr.rolling(_steps(w), min_periods=1).std()

    # (A) 휴식HR 없으면 추정치 사용
    if resting_col in df.columns:
        rest = pd.to_numeric(df[resting_col], errors='coerce')
    else:
        # 세션 중앙값-10bpm 정도의 보수적 추정
        rest = pd.Series(np.nanmedian(hr) - 10, index=df.index, dtype=float)
    df['hrr_calc'] = hr - rest

    # (B) 드리프트(슬로우 컴포넌트) 피처
    # 활동 마스크: rest+10 이상
    active = (hr >= (rest + 10)).astype(float).fillna(0.0)
    grp = (active.diff().fillna(active) != 0).cumsum()
    dt_sec = dt.fillna(med_pos)
    df['t_active_s']  = (active * dt_sec).groupby(grp).cumsum()      # 활동 시작 이후 경과시간
    hrr_pos          = df['hrr_calc'].clip(lower=0)
    df['cum_hrr']    = (hrr_pos * dt_sec).fillna(0).cumsum()         # 누적 내부부하
    df['log_t_active']  = np.log1p(df['t_active_s'])
    df['sqrt_t_active'] = np.sqrt(df['t_active_s'].clip(lower=0))

    # (4) 시차 특징
    for lag_s in (-10, -5, -2, 2, 5, 10):
        k = _steps(abs(lag_s))
        name = f'hr_lag_{lag_s:+d}s'
        df[name] = hr.shift(k) if lag_s > 0 else hr.shift(-k)
    return df


def compute_metrics_per_group(merged_clean: pd.DataFrame, y: np.ndarray, pred: np.ndarray, args) -> pd.DataFrame:
    dfm = merged_clean.copy()
    dfm['__Y__'] = y; dfm['__PRED__'] = pred
    polar_src, real_src = _find_src_columns(dfm.columns)
    if polar_src and real_src: grp_cols=[polar_src, real_src]
    elif polar_src: grp_cols=[polar_src]
    elif real_src: grp_cols=[real_src]
    else:
        g={'group':'ALL','N':int(len(dfm)),'R2':_safe_r2(dfm['__Y__'].values, dfm['__PRED__'].values),
           'MAE_Lmin':float(mean_absolute_error(dfm['__Y__'], dfm['__PRED__'])),
           'RMSE_Lmin':float(rmse(dfm['__Y__'], dfm['__PRED__'])), 'MAPE_%':float(mape(dfm['__Y__'], dfm['__PRED__']))}
        return pd.DataFrame([g])
    rows=[]
    for keys, gdf in dfm.groupby(grp_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        rec={}
        if polar_src: rec['polar_src']=str(keys[0])
        if real_src:  rec['real_src']=str(keys[-1]) if polar_src else str(keys[0])
        yv=gdf['__Y__'].values; pv=gdf['__PRED__'].values
        rec.update({'N':int(len(gdf)),'R2':_safe_r2(yv,pv),
                    'MAE_Lmin':float(mean_absolute_error(yv,pv)) if len(gdf) else float('nan'),
                    'RMSE_Lmin':float(rmse(yv,pv)) if len(gdf) else float('nan'),
                    'MAPE_%':float(mape(yv,pv)) if len(gdf) else float('nan')})
        m,_src=compute_active_mask(gdf,args)
        if m is not None and m.sum()>10:
            ya=yv[m]; pa=pv[m]; err=pa-ya
            y_mean=float(np.nanmean(ya)); p_mean=float(np.nanmean(pa))
            rec.update({'Active_N':int(ya.size),'Active_R2':_safe_r2(ya,pa),
                        'Active_MAE_Lmin':float(mean_absolute_error(ya,pa)),
                        'Active_RMSE_Lmin':float(rmse(ya,pa)),
                        'Active_MAPE_%':float(mape(ya,pa)),
                        'Active_REAL_Mean_Lmin': y_mean,
                        'Active_PRED_Mean_Lmin': p_mean,
                        'Active_Bias_Mean_Lmin': float(np.nanmean(err)),
                        'Active_Error_STD_Lmin': float(np.nanstd(err)),
                        'Active_Mean_APE_%': float(abs(p_mean - y_mean) / y_mean * 100.0) if (np.isfinite(y_mean) and y_mean!=0) else float('nan'),
                        'Active_Mask_From':_src})
        rows.append(rec)
    out=pd.DataFrame(rows)
    sort_cols=[c for c in ['polar_src','real_src'] if c in out.columns]+['N']
    asc = [True]*(len(sort_cols)-1)+[False]
    out=out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    return out

def save_per_file_metrics(out_dir: Path, df: pd.DataFrame, calibrated: bool=False):
    name_csv  = 'metrics_per_file_calibrated.csv' if calibrated else 'metrics_per_file.csv'
    name_json = 'metrics_per_file_calibrated.json' if calibrated else 'metrics_per_file.json'
    df.to_csv(out_dir / name_csv, index=False, encoding='utf-8-sig')
    with open(out_dir / name_json, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)

def save_per_file_plots(out_dir: Path, merged_clean: pd.DataFrame, y: np.ndarray, pred: np.ndarray, args, suffix="") -> int:
    dfm = merged_clean.copy(); dfm['__Y__']=y; dfm['__PRED__']=pred
    polar_src, real_src = _find_src_columns(dfm.columns)
    if not (polar_src or real_src): return 0
    grp_cols=[c for c in [polar_src,real_src] if c]
    out_dir_plots = Path(out_dir) / ('per_file_plots' + suffix); out_dir_plots.mkdir(parents=True, exist_ok=True)
    baseline_col = args.polar_vo2_col if args.polar_vo2_col and args.polar_vo2_col in dfm.columns else None
    saved=0

    for keys, gdf in dfm.groupby(grp_cols, dropna=False):
        if not isinstance(keys, tuple): keys=(keys,)
        p_name=_sanitize(keys[0]) if polar_src else None
        r_name=_sanitize(keys[-1]) if real_src else None
        tag="__".join([t for t in [p_name,r_name] if t]) or "GROUP"

        try:
            fig,ax=plt.subplots(figsize=(14,4.5))

            # === Polar 기준 액티브 마스크(경계 침식 포함) ===
            m, _src = compute_active_mask(gdf, args)  # erosion 포함
            plot_active_only = int(getattr(args, "plot_active_only", 1)) == 1

            # 플롯용 데이터프레임: 액티브만 남김
            gplot = gdf.loc[m].copy() if (plot_active_only and m is not None) else gdf

            # 전체구간을 그릴 때만 회색 음영(액티브만 자르면 음영 불필요)
            if int(args.shade_active) == 1 and (not plot_active_only) and (m is not None):
                _shade_active(ax, gdf['dt'], m, alpha=float(args.shade_alpha))

            # 라인 그리기
            ax.plot(gplot['dt'], gplot['__PRED__'], label='PRED VO₂ (L/min)', alpha=0.85, linewidth=1.5, zorder=2)
            if baseline_col:
                ax.plot(gplot['dt'], to_lmin(gplot[baseline_col], 'auto'),
                        label='POLAR VO₂ (baseline)', alpha=0.7, linewidth=1.2, zorder=1)
            ax.plot(gplot['dt'], gplot['__Y__'], label='REAL VO₂ (L/min)', linewidth=2.2, zorder=3)

            ttl=f"VO₂ – REAL vs PRED  [{p_name or ''}{' | ' if p_name and r_name else ''}{r_name or ''}]"
            ax.set_title(ttl); ax.set_xlabel('Time'); ax.set_ylabel('VO₂ (L/min)'); ax.legend(); fig.tight_layout()
            fp=out_dir_plots / f"timeseries_{tag}.png"; fig.savefig(fp, dpi=150); plt.close(fig); saved+=1
        except Exception as e:
            print(f"[WARN] per-file timeseries plot 실패: {tag} :: {e}")
    return saved

def save_artifacts(out_dir: Path, merged_clean: pd.DataFrame, y: np.ndarray, pred: np.ndarray, artifact: Dict, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_df = merged_clean.copy()
    save_df['VO2_REAL_Lmin'] = y
    save_df['VO2_PRED_Lmin'] = pred

    # ✅ 전체 길이 마스크는 별도 이름으로
    m_all, src_name = compute_active_mask(merged_clean, args)
    if m_all is not None and m_all.sum() > 10:
        ya = np.asarray(y)[m_all]
        pa = np.asarray(pred)[m_all]
        err = pa - ya
        y_mean=float(np.nanmean(ya)); p_mean=float(np.nanmean(pa))
        artifact['metrics'].update({
            'Active_N':int(ya.size),'Active_R2':_safe_r2(ya,pa),
            'Active_MAE_Lmin':float(mean_absolute_error(ya,pa)),
            'Active_RMSE_Lmin':float(rmse(ya,pa)),
            'Active_MAPE_%':float(mape(ya,pa)),
            'Active_REAL_Mean_Lmin': y_mean,
            'Active_PRED_Mean_Lmin': p_mean,
            'Active_Bias_Mean_Lmin': float(np.nanmean(err)),
            'Active_Error_STD_Lmin': float(np.nanstd(err)),
            'Active_Mean_APE_%': float(abs(p_mean - y_mean) / y_mean * 100.0) if (np.isfinite(y_mean) and y_mean!=0) else float('nan'),
            'Active_Mask_From':src_name,
            'Active_Border_Sec': float(args.active_border_sec)
        })

        # === 세션 바이어스 헤드(REAL로 잔차 학습 → HR만으로 예측) ===
     # === bias-head 학습 블록 ===
    try:
        polar_src, _ = _find_src_columns(merged_clean.columns)
        rows, targets = [], []
        if polar_src:
            for psrc, g in merged_clean.groupby(polar_src, dropna=False):
                # ✅ 세션 마스크는 세션 전용 이름
                m_sess, _ = compute_active_mask(g, args)
                if m_sess is None or m_sess.sum() < 30:
                    continue
                idx = g.index
                off = float(np.nanmean(np.asarray(y)[idx][m_sess] - np.asarray(pred)[idx][m_sess]))
                feats = _session_feats(g, hr_col=args.hr_col or 'hr')
                if np.isfinite(off):
                    rows.append(feats); targets.append(off)

        bias_head = None
        if len(rows) >= 5:
            dfH = pd.DataFrame(rows)
            impH = SimpleImputer(strategy='median')
            scH  = StandardScaler()
            XH   = scH.fit_transform(impH.fit_transform(dfH))
            mdlH = Ridge(alpha=0.5, random_state=42)
            mdlH.fit(XH, np.asarray(targets, float))
            bias_head = {
                'feature_names': list(dfH.columns),
                'imputer': impH, 'scaler': scH, 'model': mdlH
            }
            artifact['bias_head'] = bias_head

        
    except Exception as e:
        print(f"[WARN] bias-head fit failed: {e}")
    try:
        polar_src, _ = _find_src_columns(merged_clean.columns)
        rows_ab, targ_a, targ_b = [], [], []
        if polar_src:
            for psrc, g in merged_clean.groupby(polar_src, dropna=False):
                m_sess, _ = compute_active_mask(g, args)
                if m_sess is None or m_sess.sum() < 30:
                    continue
                yy = np.asarray(y)[g.index][m_sess]
                pp = np.asarray(pred)[g.index][m_sess]
                # 세션별 best affine (pp -> yy)
                if len(yy) >= 10 and np.isfinite(pp).sum() >= 10:
                    a1, b1 = np.polyfit(pp, yy, 1)
                    feats = _session_feats(g, hr_col=args.hr_col or 'hr')
                    rows_ab.append(feats); targ_a.append(float(a1)); targ_b.append(float(b1))
        affine_head = None
        if len(rows_ab) >= 5:
            dfA = pd.DataFrame(rows_ab)
            impA = SimpleImputer(strategy='median'); scA = StandardScaler()
            XA = scA.fit_transform(impA.fit_transform(dfA))
            mdl_a = Ridge(alpha=0.5, random_state=42).fit(XA, np.asarray(targ_a, float))
            mdl_b = Ridge(alpha=0.5, random_state=42).fit(XA, np.asarray(targ_b, float))
            affine_head = {'feature_names': list(dfA.columns),
                           'imputer': impA, 'scaler': scA,
                           'mdl_a': mdl_a, 'mdl_b': mdl_b}
            artifact['affine_head'] = affine_head
    except Exception as e:
        print(f"[WARN] affine-head fit failed: {e}")
    with open(out_dir / 'metrics.json','w',encoding='utf-8') as f:
        json.dump(artifact['metrics'], f, ensure_ascii=False, indent=2)
    with open(out_dir / 'model_artifact.pkl','wb') as f:
        pickle.dump(artifact,f)

    # --- 전체 타임시리즈 (액티브만 표시 옵션) ---
    try:
        fig, ax = plt.subplots(figsize=(14, 4.5))

        # Polar 기준 액티브(경계 침식 포함)
        m_plot, _src_plot = compute_active_mask(merged_clean, args)
        plot_active_only = int(getattr(args, "plot_active_only", 1)) == 1

        gplot = merged_clean.loc[m_plot].copy() if (plot_active_only and m_plot is not None) else merged_clean
        y_plot = pd.Series(y, index=merged_clean.index).loc[gplot.index].values
        pred_plot = pd.Series(pred, index=merged_clean.index).loc[gplot.index].values

        # 전체 축일 때만 액티브 음영
        if int(args.shade_active) == 1 and (not plot_active_only) and (m_plot is not None):
            _shade_active(ax, merged_clean['dt'], m_plot, alpha=float(args.shade_alpha))

        ax.plot(gplot['dt'], pred_plot, label='PRED VO₂ (L/min)', alpha=0.85, linewidth=1.5, zorder=2)

        if artifact.get('baseline_col') and artifact['baseline_col'] in merged_clean.columns:
            ax.plot(gplot['dt'], to_lmin(gplot[artifact['baseline_col']], 'auto'),
                    label='POLAR VO₂ (baseline)', alpha=0.7, linewidth=1.2, zorder=1)

        ax.plot(gplot['dt'], y_plot, label='REAL VO₂ (L/min)', linewidth=2.2, zorder=3)
        ax.set_xlabel('Time'); ax.set_ylabel('VO₂ (L/min)'); ax.set_title('VO₂ – REAL vs PRED')
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir / 'timeseries_vo2.png', dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[WARN] 타임시리즈 플롯 실패: {e}")

    # --- 스캐터(액티브만) ---
    try:
        fig2, ax2 = plt.subplots(figsize=(5.2, 5.2))
        m_scatter, _ = compute_active_mask(merged_clean, args)
        if m_scatter is not None and m_scatter.sum() > 0:
            yy = np.asarray(y)[m_scatter]
            pp = np.asarray(pred)[m_scatter]
            title = 'REAL vs PRED (ACTIVE ONLY)'
        else:
            yy = np.asarray(y)
            pp = np.asarray(pred)
            title = 'REAL vs PRED'

        ax2.scatter(yy, pp, s=12, alpha=0.6, edgecolors='none')
        mn = float(min(np.nanmin(yy), np.nanmin(pp)))
        mx = float(max(np.nanmax(yy), np.nanmax(pp)))
        ax2.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.0)
        ax2.set_xlabel('REAL VO₂ (L/min)'); ax2.set_ylabel('PRED VO₂ (L/min)')
        ax2.set_title(title)
        fig2.tight_layout(); fig2.savefig(out_dir / 'scatter_real_pred.png', dpi=150); plt.close(fig2)
    except Exception as e:
        print(f"[WARN] 스캐터 플롯 실패: {e}")

    # (선택) 스캐터 플롯(전체)
    try:
        fig2, ax2 = plt.subplots(figsize=(5.2, 5.2))
        ax2.scatter(y, pred, s=12, alpha=0.6, edgecolors='none')
        mn = float(min(np.nanmin(y), np.nanmin(pred)))
        mx = float(max(np.nanmax(y), np.nanmax(pred)))
        ax2.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.0)
        ax2.set_xlabel('REAL VO₂ (L/min)'); ax2.set_ylabel('PRED VO₂ (L/min)')
        ax2.set_title('REAL vs PRED (ALL)')
        fig2.tight_layout(); fig2.savefig(out_dir / 'scatter_real_pred_all.png', dpi=150); plt.close(fig2)

    except Exception as e:
        print(f"[WARN] 스캐터 플롯 실패: {e}")
    pred_cal, params = calibrate_predictions(merged_clean, y, pred, args)
    save_df_cal = save_df.copy()
    save_df_cal['VO2_PRED_CAL_Lmin'] = pred_cal
    save_df_cal.to_csv(out_dir / 'merged_with_pred_calibrated.csv', index=False, encoding='utf-8-sig')

    metrics_cal = dict(artifact['metrics'])
    m_cal, _ = compute_active_mask(merged_clean, args)  # 또는 m_all 재사용
    if m_cal is not None and m_cal.sum() > 10:
        ya = np.asarray(y)[m_cal]
        pa = np.asarray(pred_cal)[m_cal]
        err = pa - ya
        y_mean=float(np.nanmean(ya)); p_mean=float(np.nanmean(pa))
        metrics_cal.update({
            'Active_R2':_safe_r2(ya,pa),
            'Active_MAE_Lmin':float(mean_absolute_error(ya,pa)),
            'Active_RMSE_Lmin':float(rmse(ya,pa)),
            'Active_MAPE_%':float(mape(ya,pa)),
            'Active_REAL_Mean_Lmin': y_mean,
            'Active_PRED_Mean_Lmin': p_mean,
            'Active_Bias_Mean_Lmin': float(np.nanmean(err)),
            'Active_Error_STD_Lmin': float(np.nanstd(err)),
            'Active_Mean_APE_%': float(abs(p_mean - y_mean) / y_mean * 100.0) if (np.isfinite(y_mean) and y_mean!=0) else float('nan'),
            'Active_Mask_From': metrics_cal.get('Active_Mask_From','phase/active(POLAR)'),
            'Active_Border_Sec': float(args.active_border_sec),
            'Calibrated': args.active_calibrate
        })
    with open(out_dir / 'metrics_calibrated.json','w',encoding='utf-8') as f:
        json.dump(metrics_cal, f, ensure_ascii=False, indent=2)

def load_csvs(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"패턴에 맞는 파일이 없습니다: {glob_pat}")
    dfs=[]
    for fp in files:
        try:
            df=pd.read_csv(fp, encoding='utf-8')
        except UnicodeDecodeError:
            df=pd.read_csv(fp, encoding='cp949')
        df['__src__']=os.path.basename(fp); dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def best_lag_by_corr(t_polar, t_real, a, b, lags: List[int]) -> int:
    """
    단순 상관 기반 best lag 계산.
    - NaN/Inf는 자동 마스킹
    - 너무 짧은 구간(유효 표본 < 5)은 건너뜀
    - l>0  : REAL을 앞당긴 것과 동일하게 a[l:], b[:-l]
    - l<0  : REAL을 뒤로 민 것과 동일하게 a[:len(a)+l], b[-l:]
    """
    a = np.asarray(pd.to_numeric(a, errors='coerce'), dtype=float)
    b = np.asarray(pd.to_numeric(b, errors='coerce'), dtype=float)

    best_r, best_l = -1e9, 0
    n = min(len(a), len(b))
    if n < 5:
        return 0

    for l in lags:
        if l == 0:
            aa, bb = a, b
        elif l > 0:
            if l >= n: 
                continue
            aa, bb = a[l:], b[:-l]
        else:  # l < 0
            k = -l
            if k >= n:
                continue
            aa, bb = a[:-k], b[k:]

        # 유효값만 사용
        m = np.isfinite(aa) & np.isfinite(bb)
        if m.sum() < 5:
            continue

        try:
            r = np.corrcoef(aa[m], bb[m])[0, 1]
        except Exception:
            r = np.nan

        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, l

    return int(best_l)


def main(args):
    if args.predict_from_polar:
        predict_from_polar_only(args)
        return
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    polar = load_csvs(args.polar_glob); real = load_csvs(args.real_glob)
    polar,_ = coerce_datetime(polar, prefer_col=args.polar_time_col)
    real,_  = coerce_datetime(real, prefer_col=args.real_time_col)
    if not args.hr_col: args.hr_col = detect_hr_col(polar)
    if not args.polar_vo2_col: args.polar_vo2_col = detect_polar_vo2_col(polar)
    if not args.real_vo2_col:  args.real_vo2_col  = detect_real_vo2_col(real)
    if not args.real_vo2_col: raise ValueError("REAL(호흡가스) VO₂ 컬럼을 찾지 못했습니다. --real-vo2-col 옵션으로 지정하세요.")
    real = real.copy()
    if args.time_shift == 'manual' and abs(float(args.real_lead_sec)) > 0:
        real['dt'] = lead_seconds(real['dt'], args.real_lead_sec)
    merged = merge_time_nearest(polar, real, tol_sec=args.tolerance_sec)
    merged = add_hr_features(merged, dt_col='dt', hr_col=args.hr_col or 'hr', resting_col='restingHr')

    match_rate = merged[args.real_vo2_col].notna().mean()*100.0
    print(f"[INFO] merge_asof(nearest, ±{args.tolerance_sec}s): match_rate ≈ {match_rate:5.2f}%")
    if args.time_shift == 'auto' and int(args.apply_best_lag) == 1:
        ref_a = merged[args.hr_col] if (args.hr_col and args.hr_col in merged.columns) \
            else (merged[args.polar_vo2_col] if (args.polar_vo2_col and args.polar_vo2_col in merged.columns) else merged[args.real_vo2_col])
        lags = [int(x) for x in str(args.lags).split(',')] if (',' in str(args.lags)) else list(range(-30,31,1))
        best_lag = best_lag_by_corr(merged['dt'], merged['dt'], ref_a, merged[args.real_vo2_col], lags)
        print(f"[INFO] best lag(lead, sec) ≈ {best_lag} s (양수=REAL 앞당김)")
        real2 = real.copy(); real2['dt'] = lead_seconds(real2['dt'], best_lag)
        merged = merge_time_nearest(polar, real2, tol_sec=args.tolerance_sec)
        match_rate = merged[args.real_vo2_col].notna().mean()*100.0
        print(f"[INFO] re-merge after best-lag: match_rate ≈ {match_rate:5.2f}%")

    if len(merged) > 1:
        dt_diff = (merged['dt'].diff().dt.total_seconds()).dropna()
        med_dt = float(dt_diff.median()) if len(dt_diff) else 1.0; freq_hz = 1.0 / max(med_dt, 1e-6)
    else:
        freq_hz = 1.0
    if args.ema_sec > 0:
        num_cols=[c for c in merged.columns if np.issubdtype(merged[c].dtype, np.number)]
        for c in num_cols: merged[c]=ema(merged[c], args.ema_sec, freq_hz)
    artifact, pred, y_clean, merged_clean = train_model(merged, args)
    save_artifacts(out_dir, merged_clean, y_clean, pred, artifact, args)
    df_pf_base = compute_metrics_per_group(merged_clean, y_clean, pred, args); save_per_file_metrics(out_dir, df_pf_base, calibrated=False)
    pred_cal, _ = calibrate_predictions(merged_clean, y_clean, pred, args)
    df_pf_cal  = compute_metrics_per_group(merged_clean, y_clean, pred_cal, args); save_per_file_metrics(out_dir, df_pf_cal, calibrated=True)
    # per-file plots 저장 (요청 시)
    if int(args.save_per_file_plots) == 1:
        _n1 = save_per_file_plots(out_dir, merged_clean, y_clean, pred, args, suffix="")
        _n2 = save_per_file_plots(out_dir, merged_clean, y_clean, pred_cal, args, suffix="_calibrated")

    print("\n[Per-File Metrics] (보정 전 상위 8개)")
    show_cols=[c for c in ['polar_src','real_src','N','R2','MAE_Lmin','RMSE_Lmin','MAPE_%','Active_R2','Active_MAPE_%','Active_Mean_APE_%','Active_Bias_Mean_Lmin'] if c in df_pf_base.columns]
    print(df_pf_base[show_cols].head(8).to_string(index=False))
    print("\n[Per-File Metrics-Calibrated] (보정 후 상위 8개)")
    print(df_pf_cal[show_cols].head(8).to_string(index=False))
    print("\n[INFO] === Summary ===")
    print(f"  Files    : POLAR=({args.polar_glob}), REAL=({args.real_glob})")
    print(f"  RealLead : {args.real_lead_sec:.1f} sec (양수=REAL 앞당김)")
    print(f"  Tolerance: ±{args.tolerance_sec}s, EMA={args.ema_sec}s")
    print(f"  Residual?: {bool(args.use_residual)}  Isotonic?: {bool(args.use_isotonic)}  Calibrate?: {args.active_calibrate}")
    print(f"  Active   : source={args.active_mask_from}, border=±{args.active_border_sec}s around edges")
    if artifact.get('baseline_col', None): print(f"  Baseline : {artifact['baseline_col']} (auto-unit→L/min)")
    print("  Metrics(base)  :", json.dumps(artifact['metrics'], ensure_ascii=False, indent=2))
    print("  Outputs  : ", out_dir.resolve())
    print("             - merged_with_pred.csv / merged_with_pred_calibrated.csv")
    print("             - metrics.json / metrics_calibrated.json")
    print("             - metrics_per_file.csv / metrics_per_file_calibrated.csv")
    print("             - calibration_params.json")
    print("             - model_artifact.pkl")
    print("             - timeseries_vo2.png, scatter_real_pred.png")
    if int(args.save_per_file_plots)==1:
        print("             - per_file_plots/*.png, per_file_plots_calibrated/*.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="[Polar HR] ↔ [호흡가스 REAL VO₂] 파이프라인 (v1.9.6-patch2)")
    # ↓↓↓ required 제거
    p.add_argument("--polar-glob")
    p.add_argument("--real-glob")
    p.add_argument("--out", default="out_vo2")
    p.add_argument("--polar-time-col", default=None)
    p.add_argument("--real-time-col",  default=None)
    p.add_argument("--hr-col", default=None)
    p.add_argument("--polar-vo2-col", default=None)
    p.add_argument("--real-vo2-col",  default=None)
    p.add_argument("--real-lead-sec", type=float, default=0.0)
    p.add_argument("--tolerance-sec", type=float, default=0.5)
    p.add_argument("--ema-sec", type=float, default=0.0)
    p.add_argument("--real-unit", choices=['auto','ml','l'], default='auto')
    p.add_argument("--apply-best-lag", type=int, default=0)
    p.add_argument("--lags", default="-30:31:1")
    p.add_argument("--use-residual", type=int, default=1)
    p.add_argument("--use-isotonic", type=int, default=1)
    p.add_argument("--model", choices=['ridge','huber'], default='ridge')
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--active-only", type=int, default=0)
    p.add_argument("--rest-hr", type=float, default=0.0)
    p.add_argument("--active-hr-delta", type=float, default=10.0)
    p.add_argument("--save-per-file-plots", type=int, default=1)
    p.add_argument("--shade-active", type=int, default=1)
    p.add_argument("--shade-alpha", type=float, default=0.12)
    p.add_argument("--active-mask-from", choices=['polar','hr','auto'], default='polar')
    p.add_argument("--phase-active-values", type=str, default="active,work,exercise")
    p.add_argument("--active-calibrate", choices=['none','bias','affine','meanvar'], default='none')
    p.add_argument("--active-calib-trim-pct", type=float, default=0.0)
    p.add_argument("--active-border-sec", type=float, default=0.0)
    p.add_argument("--plot-active-only", type=int, default=1)
    p.add_argument("--predict-from-polar", default=None)
    p.add_argument("--load-artifact", default=None)
    p.add_argument("--cat-cols", default="",
               help="원-핫 인코딩할 범주형 열들(콤마 구분). 예: 'motion_id,sex'")
    p.add_argument("--apply-calibration-from", default=None,
               help="훈련에서 저장된 calibration_params.json 경로")
    p.add_argument("--calibrate-active-only", type=int, default=1,
                help="1=ACTIVE 구간에만 보정 적용, 0=전체 적용")
    p.add_argument("--iso-blend-w", type=float, default=0.5,
              help="Isotonic 출력과 pre-ISO 예측을 가중합(0~1). 1=ISO만, 0=pre-ISO만")
    p.add_argument("--use-bias-head", type=int, default=1,
                help="1=HR-only 세션 오프셋 보정 사용, 0=미사용. calibration_params가 있으면 bias-head는 자동 비활성.")
    p.add_argument("--calib-fallback", choices=['none','global'], default='none',
               help="per-file 캘리브 키가 없을 때 GLOBAL을 쓸지 여부")
    p.add_argument("--bias-head-shrink", type=float, default=0.5,
                help="bias-head 오프셋 축소 계수(0~1)")
    p.add_argument("--bias-head-max-abs", type=float, default=0.15,
                help="캘리브 없을 때 bias-head 최대 절대값(L/min)")
    p.add_argument("--bias-head-max-abs-after-cal", type=float, default=0.05,
                help="캘리브 있는 세션에서 bias-head 최대 절대값(L/min)")
    p.add_argument("--force-global-calib", type=int, default=0,
               help="1이면 calibration_params.json의 __GLOBAL__만 항상 적용(파일명 무시)")
    p.add_argument("--use-affine-head", type=int, default=0,
                help="1이면 세션별 스케일/오프셋 자동 보정(a,b) 적용")
    p.add_argument("--affine-head-shrink-a", type=float, default=0.5)
    p.add_argument("--affine-head-shrink-b", type=float, default=0.5)
    p.add_argument("--affine-head-max-scale-dev", type=float, default=0.25,
                help="a는 1±이 값으로 클램프")
    p.add_argument("--affine-head-max-abs-b", type=float, default=0.12,
                help="세션 오프셋 b의 절대 상한(L/min)")
    p.add_argument("--time-shift", choices=['none','manual','auto'], default='none',
               help="시간 이동/정렬 모드: none=금지, manual=real-lead-sec 만큼만, auto=best-lag 사용")
    p.add_argument("--ref-table", default=None, help="사용자/동작별 타겟 VO2(ml)가 정의된 CSV 파일 경로")

    args = p.parse_args()
    args.phase_active_values = [s.strip() for s in (args.phase_active_values or '').split(',') if s.strip()]
    args.cat_cols = [s.strip() for s in (getattr(args, 'cat_cols', '') or '').split(',') if s.strip()]
    
    # [NEW] 참조 테이블 경로 인자 추가

    # ★ 모드별 유효성 검사: 예측 모드가 아니면 polar/real 둘 다 필수
    if not args.predict_from_polar:
        if not args.polar_glob or not args.real_glob:
            p.error("--polar-glob 와 --real-glob 둘 다 필요합니다 (훈련 모드). "
                    "예측만 하려면 --predict-from-polar 를 사용하세요.")

    main(args)
