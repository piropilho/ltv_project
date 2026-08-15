"""
트랙 B: XGBoost — "다음에 무슨 일이 일어날지" 예측하는 시간인지형 예측모델
=====================================================================
입력: data/master_19_25_cleaning.csv (실거래 데이터 단독. 학군/지하철 등 입지변수는
      아직 결합하지 않음 — XGBoost_기획안_v2.md 1절)

설계 원칙 (XGBoost_기획안_v2.md 반영):
- 관측 단위: (complex_id, as_of_month) 패널. 개별 단지 월별 거래량이 희박하므로,
  가격 관련 피처/타겟은 전부 "단지 + k개 최근접 이웃"의 트레일링 평균으로 스무딩한다.
  k=N_NEIGHBORS=5는 spatial_autocorrelation.py의 상관로그램 분석으로 근거를 마련한
  값이며 GWR 트랙(gwr_analysis.py)과 동일하게 맞췄다. 단, 이 k는 "가격" 기준으로 구한
  값이므로 향후 인구/교통 등 새 변수가 들어오면 그 변수 자체로 별도 재검증이 필요하다
  (XGBoost_기획안_v2.md 2절).
- 타겟: t 시점 대비 t+HORIZON_MONTHS 시점의 지역 스무딩 가격 변화율(%) — 회귀.
- 피처는 전부 t 시점까지의 트레일링 정보만 사용 (미래 누수 없음). 타겟만 미래를 본다.
- 검증: walk-forward(확장 윈도우)만 사용, random split 금지. 폴드 경계는 달력 균등분할이
  아니라 fold_boundary_design.py가 가격 시계열만으로 탐지한 국면 전환점을 사용 (피처 테스트
  결과와 무관하게 결정됨). 가장 마지막 국면은 폴드 구성에 전혀 사용하지 않고 최종 1회
  평가에만 쓴다.
- 정책 충격 변수는 개별 이벤트 더미가 아니라 일반화된 형태
  (days_since_last_tightening, n_tightening_past_1y)로 정의 — eda_7years.py의
  POLICY_EVENTS를 재사용해 GWR/EDA와 하나의 출처를 공유한다.
- 모든 튜닝 가능한 값(윈도우 길이, k, 호라이즌, 폴드 구성 등)은 아래 상수로 노출한다.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree
import xgboost as xgb

from eda_7years import POLICY_EVENTS

# ------------------------------------------------------------------
# 파라미터 (값만 바꿔서 재실행 가능 — 지금은 베이스라인 기본값)
# ------------------------------------------------------------------
MASTER_CSV = "data/master_19_25_cleaning.csv"

N_NEIGHBORS = 5              # 이웃 집계 단지 수 (gwr_analysis.py / spatial_autocorrelation.py와 동일 근거)
SMOOTH_WINDOW_MONTHS = 3     # 현재 시점 지역 가격수준 계산에 쓰는 트레일링 개월 수
MOMENTUM_LAG_MONTHS = 3      # 모멘텀 비교 기준 시차 (t 대비 t-LAG 스무딩값)
HORIZON_MONTHS = 6           # 예측 호라이즌 (타겟: t -> t+HORIZON 가격변화율)
MIN_TRADES_IN_WINDOW = 3     # 스무딩 윈도우 내 (단지+이웃) 최소 거래건수, 미달 시 결측 처리

EXPANDING_WINDOW = True      # True=확장윈도우, False=롤링윈도우(추후 실험용, 현재 미구현)
MIN_TRAIN_MONTHS = 24        # 국면(폴드)이 테스트 폴드로 채택되려면 그 이전에 필요한 최소 학습 개월 수
FOLD_BOUNDARIES_CSV = "data/fold_boundaries.csv"  # fold_boundary_design.py 산출물 (국면 경계)

DAYS_SINCE_NO_EVENT_SENTINEL = 9999  # as_of 시점 이전에 정책 이벤트가 한 번도 없었을 때
N_TIGHTENING_LOOKBACK_DAYS = 365     # n_tightening_past_1y 집계 기간

BASE_RATE_CSV = "data/base_rate_2019_present.csv"  # 기준금리_크롤링.py 산출물
RATE_CHANGE_LOOKBACK_MONTHS = 3      # 기준금리 변화량(%p) 계산에 쓰는 트레일링 개월 수
RATE_CHANGE_THRESHOLD = 0.3          # 3개월 내 1스텝(0.25%p)까지는 노이즈로 보고 0 처리 (게이팅 버전)

SDI_CSV = "data/supply_demand_index_2019_present.csv"  # 매매수급동향_크롤링.py 산출물
SDI_REGION = "dongbuk"                # 동북권 (동대문구가 속한 권역, 구 단위 데이터는 미제공)
SDI_CHANGE_LOOKBACK_MONTHS = 3        # 지수 변화량 계산에 쓰는 트레일링 개월 수
SDI_CHANGE_THRESHOLD = 10             # 3개월 변화량 표준편차(~11.9)에 근접한 값. 이보다 작은
                                       # 변화는 노이즈로 보고 0으로 처리 (게이팅 버전 전용)

EARTH_RADIUS_KM = 6371.0

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
)

FEATURE_COLS = [
    "excluUseAr", "floor", "building_age",
    "local_price_level", "local_momentum",
    "days_since_last_tightening", "n_tightening_past_1y",
]
TARGET_COL = "future_price_change_pct"


# ------------------------------------------------------------------
# 1. 데이터 로드 + 단지 정적 테이블 + 이웃 매핑
# ------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["year_month"] = df["contract_date"].dt.to_period("M")
    return df


def build_complex_static(df: pd.DataFrame) -> pd.DataFrame:
    """단지별 정적 속성 (좌표, 전용면적/층 평균, 준공연도) — building_age는 as_of 시점에 동적으로 계산"""
    table = (
        df.groupby("complex_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"),
             excluUseAr=("excluUseAr", "mean"), floor=("floor", "mean"),
             buildYear=("buildYear", "first"))
        .reset_index()
    )
    return table


def build_neighbor_map(complex_static: pd.DataFrame, k: int = N_NEIGHBORS) -> dict:
    """단지별 최근접 이웃 k개 (자기 자신 제외) — gwr_analysis.py add_neighbor_features와 동일 방식"""
    coords_rad = np.radians(complex_static[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    k_query = min(k + 1, len(complex_static))
    _, idx = tree.query(coords_rad, k=k_query)

    ids = complex_static["complex_id"].values
    neighbor_map = {}
    for i, cid in enumerate(ids):
        neighbor_ids = [ids[j] for j in idx[i] if j != i][:k]
        neighbor_map[cid] = neighbor_ids
    return neighbor_map


# ------------------------------------------------------------------
# 2. 월별 (단지 x 월) 집계 행렬 — 트레일링 윈도우 조회를 빠르게 하기 위한 누적합 구조
# ------------------------------------------------------------------
def build_monthly_matrices(df: pd.DataFrame, complex_static: pd.DataFrame):
    """
    sum_price_mat, n_mat: (n_complex x n_month) 배열. [c, m] = 해당 단지의 해당 월 가격합/거래건수.
    반환되는 누적합(cumsum)으로 임의 구간 [t1, t2]의 합을 O(1)에 조회할 수 있다.
    """
    all_months = pd.period_range(df["year_month"].min(), df["year_month"].max(), freq="M")
    month_idx = {m: i for i, m in enumerate(all_months)}
    complex_idx = {cid: i for i, cid in enumerate(complex_static["complex_id"].values)}

    n_complex, n_month = len(complex_idx), len(all_months)
    sum_price = np.zeros((n_complex, n_month))
    n_trades = np.zeros((n_complex, n_month))

    monthly = df.groupby(["complex_id", "year_month"])["price_per_m2"].agg(["sum", "size"]).reset_index()
    for row in monthly.itertuples(index=False):
        ci, mi = complex_idx[row.complex_id], month_idx[row.year_month]
        sum_price[ci, mi] = row.sum
        n_trades[ci, mi] = row.size

    cumsum_price = np.cumsum(sum_price, axis=1)
    cumsum_n = np.cumsum(n_trades, axis=1)
    return all_months, month_idx, complex_idx, cumsum_price, cumsum_n


def window_sum(cumsum: np.ndarray, complex_ids: list, ci_map: dict, m_start: int, m_end: int) -> float:
    """[m_start, m_end] (양끝 포함, 0-index) 구간의 여러 단지 합산값. 범위 밖이면 0."""
    if m_start > m_end:
        return 0.0
    total = 0.0
    for cid in complex_ids:
        ci = ci_map[cid]
        hi = cumsum[ci, m_end]
        lo = cumsum[ci, m_start - 1] if m_start > 0 else 0.0
        total += hi - lo
    return total


# ------------------------------------------------------------------
# 3. 정책 충격 피처 (일반화된 형태 — eda_7years.POLICY_EVENTS 재사용)
# ------------------------------------------------------------------
def policy_shock_features(as_of_date: pd.Timestamp) -> tuple[float, int]:
    event_dates = sorted(pd.to_datetime(list(POLICY_EVENTS.values())))
    past_events = [d for d in event_dates if d <= as_of_date]

    if not past_events:
        days_since = DAYS_SINCE_NO_EVENT_SENTINEL
    else:
        days_since = (as_of_date - past_events[-1]).days

    window_start = as_of_date - pd.Timedelta(days=N_TIGHTENING_LOOKBACK_DAYS)
    n_recent = sum(window_start < d <= as_of_date for d in event_dates)
    return days_since, n_recent


# ------------------------------------------------------------------
# 4a. 선행지표 — 기준금리 (기준금리_크롤링.py 산출물)
# ------------------------------------------------------------------
def load_base_rate() -> pd.Series:
    """year_month(Period) -> base_rate(%) 시리즈. 트레일링 조회만 하므로 미래 누수 없음."""
    df = pd.read_csv(BASE_RATE_CSV)
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df.set_index("year_month")["base_rate"]


def base_rate_features(rate: pd.Series, as_of_month: pd.Period) -> tuple[float, float, float]:
    """as_of 시점 기준금리 수준(level), 3개월 변화량(change_raw), 노이즈 게이팅한 변화량(change_gated)"""
    level = rate.get(as_of_month, np.nan)
    lag_month = as_of_month - RATE_CHANGE_LOOKBACK_MONTHS
    level_lag = rate.get(lag_month, np.nan)
    if np.isnan(level) or np.isnan(level_lag):
        return level, np.nan, np.nan
    change_raw = level - level_lag
    change_gated = change_raw if abs(change_raw) > RATE_CHANGE_THRESHOLD else 0.0
    return level, change_raw, change_gated


# ------------------------------------------------------------------
# 4b. 선행지표 — 매매수급동향지수 (매매수급동향_크롤링.py 산출물)
# ------------------------------------------------------------------
def load_supply_demand_index() -> pd.Series:
    """year_month(Period) -> supply_demand_index 시리즈. 트레일링 조회만 하므로 미래 누수 없음."""
    df = pd.read_csv(SDI_CSV)
    df = df[df["series"] == SDI_REGION]
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    return df.set_index("year_month")["supply_demand_index"]


def supply_demand_features(sdi: pd.Series, as_of_month: pd.Period) -> tuple[float, float, float]:
    """as_of 시점 지수 수준(level), 3개월 변화량(change_raw), 노이즈 게이팅한 변화량(change_gated)"""
    level = sdi.get(as_of_month, np.nan)
    lag_month = as_of_month - SDI_CHANGE_LOOKBACK_MONTHS
    level_lag = sdi.get(lag_month, np.nan)
    if np.isnan(level) or np.isnan(level_lag):
        return level, np.nan, np.nan
    change_raw = level - level_lag
    change_gated = change_raw if abs(change_raw) > SDI_CHANGE_THRESHOLD else 0.0
    return level, change_raw, change_gated


# ------------------------------------------------------------------
# 5. 패널 구성 — (complex_id, as_of_month) 단위로 피처/타겟 생성
# ------------------------------------------------------------------
def build_panel(df: pd.DataFrame, complex_static: pd.DataFrame, neighbor_map: dict,
                 rate: pd.Series, sdi: pd.Series) -> pd.DataFrame:
    all_months, month_idx, ci_map, cs_price, cs_n = build_monthly_matrices(df, complex_static)
    n_month = len(all_months)

    static_lookup = complex_static.set_index("complex_id")[["excluUseAr", "floor", "buildYear"]]

    def local_level(cid: int, m: int) -> tuple[float, float]:
        """단지+이웃 그룹의 [m-SMOOTH_WINDOW+1, m] 구간 스무딩 가격수준. (level, n_trades) 반환"""
        group = [cid] + neighbor_map[cid]
        lo = m - SMOOTH_WINDOW_MONTHS + 1
        if lo < 0:
            return np.nan, 0
        total_price = window_sum(cs_price, group, ci_map, lo, m)
        total_n = window_sum(cs_n, group, ci_map, lo, m)
        if total_n < MIN_TRADES_IN_WINDOW:
            return np.nan, total_n
        return total_price / total_n, total_n

    rows = []
    for cid in complex_static["complex_id"].values:
        build_year = static_lookup.loc[cid, "buildYear"]
        excl = static_lookup.loc[cid, "excluUseAr"]
        floor = static_lookup.loc[cid, "floor"]

        for m in range(n_month):
            as_of_month = all_months[m]

            level_now, n_now = local_level(cid, m)
            level_lag, _ = local_level(cid, m - MOMENTUM_LAG_MONTHS)
            future_m = m + HORIZON_MONTHS
            level_future, n_future = (np.nan, 0) if future_m >= n_month else local_level(cid, future_m)

            if np.isnan(level_now) or np.isnan(level_future) or level_now == 0:
                continue
            momentum = np.nan if (np.isnan(level_lag) or level_lag == 0) else (level_now / level_lag - 1) * 100

            as_of_date = as_of_month.to_timestamp(how="end")
            days_since, n_recent = policy_shock_features(as_of_date)
            rate_level, rate_change_raw, rate_change_gated = base_rate_features(rate, as_of_month)
            sdi_level, sdi_change_raw, sdi_change_gated = supply_demand_features(sdi, as_of_month)

            rows.append({
                "complex_id": cid,
                "as_of_month": as_of_month,
                "excluUseAr": excl,
                "floor": floor,
                "building_age": as_of_month.year - build_year,
                "local_price_level": level_now,
                "local_momentum": momentum,
                "days_since_last_tightening": days_since,
                "n_tightening_past_1y": n_recent,
                "base_rate_level": rate_level,
                "base_rate_change_raw": rate_change_raw,
                "base_rate_change_gated": rate_change_gated,
                "sdi_level": sdi_level,
                "sdi_change_raw": sdi_change_raw,
                "sdi_change_gated": sdi_change_gated,
                "n_trades_now": n_now,
                TARGET_COL: (level_future / level_now - 1) * 100,
            })

    panel = pd.DataFrame(rows)
    print(f"[패널] {len(panel)}행 (단지 {panel['complex_id'].nunique()}개 x "
          f"as_of_month {panel['as_of_month'].nunique()}개월), "
          f"모멘텀 결측 {panel['local_momentum'].isna().sum()}행 제외 예정")
    panel = panel.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"[패널] 결측 제거 후 {len(panel)}행")
    return panel


# ------------------------------------------------------------------
# 5. walk-forward 폴드 구성 — 국면 전환점(fold_boundary_design.py) 기반 + 최종 홀드아웃 분리
# ------------------------------------------------------------------
def make_walkforward_folds(panel: pd.DataFrame):
    """
    fold_boundary_design.py가 가격 시계열만으로(피처 테스트 결과와 무관하게) 탐지한 국면
    구간을 그대로 폴드로 사용한다. 각 구간을 테스트 폴드로 쓰려면 그 구간 시작 시점까지
    MIN_TRAIN_MONTHS 이상의 선행 학습 데이터가 있어야 하며, 없으면 스킵(학습 구간에는 포함).
    마지막 구간은 항상 폴드 구성에서 제외하고 최종 1회 평가 전용 홀드아웃으로 분리한다.
    """
    segments = pd.read_csv(FOLD_BOUNDARIES_CSV)
    segments["start"] = pd.PeriodIndex(segments["start"], freq="M")
    segments["end"] = pd.PeriodIndex(segments["end"], freq="M")

    panel_months = sorted(panel["as_of_month"].unique())
    segments = segments[segments["end"] >= panel_months[0]].reset_index(drop=True)
    if len(segments) < 2:
        raise ValueError("패널 범위와 겹치는 국면 구간이 2개 미만이라 홀드아웃을 분리할 수 없습니다.")

    holdout_seg = segments.iloc[-1]
    holdout_months = [m for m in panel_months if holdout_seg["start"] <= m <= holdout_seg["end"]]

    folds = []
    for _, seg in segments.iloc[:-1].iterrows():
        train_months = [m for m in panel_months if m < seg["start"]]
        test_months = [m for m in panel_months if seg["start"] <= m <= seg["end"]]
        if len(train_months) < MIN_TRAIN_MONTHS or not test_months:
            continue
        folds.append((train_months, test_months))

    usable_months = [m for m in panel_months if m not in holdout_months]

    print(f"[국면 기반 walk-forward] 패널과 겹치는 국면 {len(segments)}개 중 마지막 국면을 "
          f"최종 홀드아웃으로 분리: {holdout_seg['start']}~{holdout_seg['end']} "
          f"(패널 기준 {len(holdout_months)}개월)")
    for i, (tr, te) in enumerate(folds, 1):
        print(f"  폴드{i}: train {tr[0]}~{tr[-1]} ({len(tr)}개월) -> "
              f"test {te[0]}~{te[-1]} ({len(te)}개월)")

    return folds, usable_months, holdout_months


# ------------------------------------------------------------------
# 6. 학습/평가
# ------------------------------------------------------------------
def fit_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list = FEATURE_COLS) -> dict:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[TARGET_COL])
    pred = model.predict(test_df[feature_cols])

    mae = mean_absolute_error(test_df[TARGET_COL], pred)
    r2 = r2_score(test_df[TARGET_COL], pred)
    return {"model": model, "mae": mae, "r2": r2, "n_train": len(train_df), "n_test": len(test_df)}


def run_walkforward(panel: pd.DataFrame, folds: list, feature_cols: list = FEATURE_COLS) -> pd.DataFrame:
    panel = panel.dropna(subset=feature_cols + [TARGET_COL])
    results = []
    for i, (train_months, test_months) in enumerate(folds, 1):
        train_df = panel[panel["as_of_month"].isin(train_months)]
        test_df = panel[panel["as_of_month"].isin(test_months)]
        r = fit_and_eval(train_df, test_df, feature_cols)
        print(f"  폴드{i}: n_train={r['n_train']:5d}  n_test={r['n_test']:4d}  "
              f"MAE={r['mae']:.2f}%p  R2={r['r2']:.3f}")
        results.append({"fold": i, "n_train": r["n_train"], "n_test": r["n_test"],
                         "mae": r["mae"], "r2": r["r2"]})
    return pd.DataFrame(results)


def run_final_holdout(panel: pd.DataFrame, usable_months: list, holdout_months: list,
                       feature_cols: list = FEATURE_COLS) -> dict:
    panel = panel.dropna(subset=feature_cols + [TARGET_COL])
    train_df = panel[panel["as_of_month"].isin(usable_months)]
    test_df = panel[panel["as_of_month"].isin(holdout_months)]
    if test_df.empty:
        print("[최종 홀드아웃] 평가할 데이터가 없음 (건너뜀)")
        return {}

    r = fit_and_eval(train_df, test_df, feature_cols)
    print(f"\n[최종 홀드아웃 평가] n_train={r['n_train']}  n_test={r['n_test']}  "
          f"MAE={r['mae']:.2f}%p  R2={r['r2']:.3f}")

    importance = pd.Series(r["model"].feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  피처 중요도:")
    for name, val in importance.items():
        print(f"    {name:28s} {val:.3f}")
    return r


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def weighted_wf_metrics(fold_results: pd.DataFrame) -> tuple[float, float]:
    """폴드별 결과를 test 표본크기로 가중평균한 MAE/R2 (참고용 — 최종 채택 판단 기준은 아님)"""
    n = fold_results["n_test"]
    w_mae = (fold_results["mae"] * n).sum() / n.sum()
    w_r2 = (fold_results["r2"] * n).sum() / n.sum()
    return w_mae, w_r2


def compare_candidates(panel: pd.DataFrame, folds: list, candidates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    후보별 폴드별 R2를 나란히 놓은 와이드 테이블 + 후보별 요약(가중평균, 베이스라인 대비
    개선된 폴드 수)을 반환한다. 판단은 가중평균 하나가 아니라 '폴드 대부분에서 고르게
    개선되는가'(승 폴드 수가 과반)를 기준으로 한다.
    """
    per_fold = {}
    summary = []
    for name, cols in candidates.items():
        print(f"\n=== walk-forward: {name} ===")
        fold_results = run_walkforward(panel, folds, cols)
        w_mae, w_r2 = weighted_wf_metrics(fold_results)
        print(f"  -> wf 가중평균 MAE={w_mae:.3f}%p  wf 가중평균 R2={w_r2:.3f}")
        per_fold[name] = fold_results.set_index("fold")["r2"]
        summary.append({"config": name, "wf_mae": w_mae, "wf_r2": w_r2})

    wide = pd.DataFrame(per_fold)
    baseline_col = wide.columns[0]
    n_folds = len(wide)

    summary_df = pd.DataFrame(summary)
    wins = []
    for name in wide.columns:
        n_improved = int((wide[name] > wide[baseline_col]).sum()) if name != baseline_col else None
        wins.append(n_improved)
    summary_df["folds_improved_vs_baseline"] = [f"{w}/{n_folds}" if w is not None else "-" for w in wins]
    summary_df["majority_improved"] = [
        (w is not None) and (w > n_folds / 2) for w in wins
    ]

    return wide, summary_df


def main():
    df = load_master()
    complex_static = build_complex_static(df)
    neighbor_map = build_neighbor_map(complex_static)
    rate = load_base_rate()
    sdi = load_supply_demand_index()

    panel = build_panel(df, complex_static, neighbor_map, rate, sdi)
    panel.to_csv("data/xgb_panel.csv", index=False, encoding="utf-8-sig")
    print("[저장] 패널 -> data/xgb_panel.csv")

    folds, usable_months, holdout_months = make_walkforward_folds(panel)

    candidates = {
        "베이스라인": FEATURE_COLS,
        "+ base_rate_level": FEATURE_COLS + ["base_rate_level"],
        "+ base_rate_change_raw": FEATURE_COLS + ["base_rate_change_raw"],
        "+ base_rate_change_gated": FEATURE_COLS + ["base_rate_change_gated"],
        "+ sdi_level": FEATURE_COLS + ["sdi_level"],
        "+ sdi_change_raw": FEATURE_COLS + ["sdi_change_raw"],
        "+ sdi_change_gated": FEATURE_COLS + ["sdi_change_gated"],
    }

    wide, summary_df = compare_candidates(panel, folds, candidates)

    print("\n=== 폴드별 R2 (후보 간 나란히 비교) ===")
    print(wide.to_string())
    wide.to_csv("data/xgb_fold_comparison.csv", encoding="utf-8-sig")

    print("\n=== 비교 요약 (판단 기준: 폴드 과반에서 베이스라인보다 R2가 개선돼야 채택 후보) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("data/xgb_candidate_summary.csv", index=False, encoding="utf-8-sig")
    print("[저장] -> data/xgb_fold_comparison.csv, data/xgb_candidate_summary.csv")

    adopted = summary_df[summary_df["majority_improved"]]
    if adopted.empty:
        print("\n[판단] 폴드 과반에서 개선된 후보 없음 -> 전부 폐기, 다음 후보(미분양 물량 등)로")
    else:
        print("\n[판단] 채택 후보:")
        for _, row in adopted.iterrows():
            print(f"  - {row['config']} (개선 폴드 {row['folds_improved_vs_baseline']}, "
                  f"wf 가중 R2 {row['wf_r2']:.3f})")

    run_final_holdout(panel, usable_months, holdout_months, FEATURE_COLS)


if __name__ == "__main__":
    main()
