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
- 검증: walk-forward(확장 윈도우)만 사용, random split 금지. 가장 최근
  FINAL_HOLDOUT_MONTHS개월은 폴드 구성에 전혀 사용하지 않고 최종 1회 평가에만 쓴다.
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
FINAL_HOLDOUT_MONTHS = 6     # 가장 최근 N개월: 폴드 구성에서 완전히 제외, 최종 1회 평가 전용
N_WALKFORWARD_FOLDS = 4      # 홀드아웃 이전 구간에서 확장윈도우 폴드 수
MIN_TRAIN_MONTHS = 24        # 첫 폴드가 확보해야 하는 최소 학습 개월 수

DAYS_SINCE_NO_EVENT_SENTINEL = 9999  # as_of 시점 이전에 정책 이벤트가 한 번도 없었을 때
N_TIGHTENING_LOOKBACK_DAYS = 365     # n_tightening_past_1y 집계 기간

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
# 4. 패널 구성 — (complex_id, as_of_month) 단위로 피처/타겟 생성
# ------------------------------------------------------------------
def build_panel(df: pd.DataFrame, complex_static: pd.DataFrame, neighbor_map: dict) -> pd.DataFrame:
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
# 5. walk-forward 폴드 구성 (확장윈도우) + 최종 홀드아웃 분리
# ------------------------------------------------------------------
def make_walkforward_folds(panel: pd.DataFrame):
    months = sorted(panel["as_of_month"].unique())
    n_total = len(months)

    n_holdout = min(FINAL_HOLDOUT_MONTHS, n_total - MIN_TRAIN_MONTHS)
    holdout_months = months[n_total - n_holdout:]
    usable_months = months[: n_total - n_holdout]

    remaining = len(usable_months) - MIN_TRAIN_MONTHS
    if remaining < N_WALKFORWARD_FOLDS:
        raise ValueError(
            f"walk-forward 폴드를 구성하기엔 데이터가 부족합니다 "
            f"(usable_months={len(usable_months)}, MIN_TRAIN_MONTHS={MIN_TRAIN_MONTHS})"
        )
    test_block = remaining // N_WALKFORWARD_FOLDS

    folds = []
    for i in range(N_WALKFORWARD_FOLDS):
        train_end = MIN_TRAIN_MONTHS + i * test_block
        test_start = train_end
        test_end = len(usable_months) if i == N_WALKFORWARD_FOLDS - 1 else train_end + test_block

        train_months = usable_months[:train_end]
        test_months = usable_months[test_start:test_end]
        if not test_months:
            continue
        folds.append((train_months, test_months))

    print(f"[walk-forward] 전체 {n_total}개월 중 최종 홀드아웃 {len(holdout_months)}개월 "
          f"({holdout_months[0]}~{holdout_months[-1] if holdout_months else '-'}) 분리")
    for i, (tr, te) in enumerate(folds, 1):
        print(f"  폴드{i}: train {tr[0]}~{tr[-1]} ({len(tr)}개월) -> "
              f"test {te[0]}~{te[-1]} ({len(te)}개월)")

    return folds, usable_months, holdout_months


# ------------------------------------------------------------------
# 6. 학습/평가
# ------------------------------------------------------------------
def fit_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
    pred = model.predict(test_df[FEATURE_COLS])

    mae = mean_absolute_error(test_df[TARGET_COL], pred)
    r2 = r2_score(test_df[TARGET_COL], pred)
    return {"model": model, "mae": mae, "r2": r2, "n_train": len(train_df), "n_test": len(test_df)}


def run_walkforward(panel: pd.DataFrame, folds: list) -> pd.DataFrame:
    results = []
    for i, (train_months, test_months) in enumerate(folds, 1):
        train_df = panel[panel["as_of_month"].isin(train_months)]
        test_df = panel[panel["as_of_month"].isin(test_months)]
        r = fit_and_eval(train_df, test_df)
        print(f"  폴드{i}: n_train={r['n_train']:5d}  n_test={r['n_test']:4d}  "
              f"MAE={r['mae']:.2f}%p  R2={r['r2']:.3f}")
        results.append({"fold": i, "n_train": r["n_train"], "n_test": r["n_test"],
                         "mae": r["mae"], "r2": r["r2"]})
    return pd.DataFrame(results)


def run_final_holdout(panel: pd.DataFrame, usable_months: list, holdout_months: list) -> dict:
    train_df = panel[panel["as_of_month"].isin(usable_months)]
    test_df = panel[panel["as_of_month"].isin(holdout_months)]
    if test_df.empty:
        print("[최종 홀드아웃] 평가할 데이터가 없음 (건너뜀)")
        return {}

    r = fit_and_eval(train_df, test_df)
    print(f"\n[최종 홀드아웃 평가] n_train={r['n_train']}  n_test={r['n_test']}  "
          f"MAE={r['mae']:.2f}%p  R2={r['r2']:.3f}")

    importance = pd.Series(r["model"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  피처 중요도:")
    for name, val in importance.items():
        print(f"    {name:28s} {val:.3f}")
    return r


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    df = load_master()
    complex_static = build_complex_static(df)
    neighbor_map = build_neighbor_map(complex_static)

    panel = build_panel(df, complex_static, neighbor_map)
    panel.to_csv("data/xgb_panel.csv", index=False, encoding="utf-8-sig")
    print("[저장] 패널 -> data/xgb_panel.csv")

    folds, usable_months, holdout_months = make_walkforward_folds(panel)

    print("\n=== walk-forward 폴드별 성능 (확장윈도우) ===")
    fold_results = run_walkforward(panel, folds)
    fold_results.to_csv("data/xgb_walkforward_results.csv", index=False, encoding="utf-8-sig")
    print("[저장] 폴드별 결과 -> data/xgb_walkforward_results.csv")

    run_final_holdout(panel, usable_months, holdout_months)


if __name__ == "__main__":
    main()
