"""
트랙 A: GWR(지리적 가중회귀) — "왜 지금 이 단지가 위험한가"를 설명하는 정적 스냅샷 분석
=======================================================================================
입력: data/csv/master_19_25_cleaning.csv (7개년, 15,365건, 316개 단지, 좌표 포함)
      + data/csv/final.csv의 정적 입지변수(경사도/학군/교통) 중 gwr_feature_selection.py가
        중첩 K-fold LASSO로 선별한 변수만 추가 결합

목적: "실거래가 데이터만으로 GWR을 돌렸을 때"와 "여기에 공간입지특성 변수를 추가했을 때"의
      설명력(Global R2)을 비교해, "공간입지특성을 추가하면 설명력이 올라간다"는 가설을 검증.
      GWR 자체는 예측모델이 아니라 "지금까지의 데이터로 오늘 시점의 공간적 리스크 패턴을
      설명"하는 게 목적이므로, 이 비교는 (LASSO 스크리닝 때와 달리) 검증셋 분리 없이
      전체표본 적합 R2로 판단한다. 다만 변수를 추가하면 R2가 기계적으로 오르는 경향이 있어,
      파라미터 수를 보정한 조정R2(adjusted R2)를 함께 본다.

설계 원칙 (기획안 반영):
- 관측 단위: 좌표점(단지) 1개 = 1행, 정적 스냅샷 (거래 단위 아님)
- 종속변수: MDD(최대낙폭)를 주력 지표로, CV(변동계수)를 보조지표로 병행 산출
  (담보가치 리스크는 "얼마나 흩어지는가"보다 "고점 대비 얼마나 빠지는가"가 은행 관점에 더 부합)
- 최소거래건수 임계값은 고정하지 않고, 커버리지 트레이드오프 표를 먼저 보여준 뒤 결정
- 검증은 없음(전체표본 회귀) — 국지적 유의성(t-value)으로 판단
- 좌표는 mgwr 규약에 따라 (lon, lat) 순서로 입력, 독립변수는 표준화
- 추가되는 입지변수는 gwr_feature_selection.py의 스크리닝 결과만 사용 (경사도 4개 반경을
  전부 넣지 않고, 중첩 K-fold로 반경/변수를 미리 골라둔 것 — 다중공선성 방지)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

MASTER_CSV = "data/csv/master_19_25_cleaning.csv"
FINAL_CSV = "data/csv/final.csv"  # gwr_feature_selection.py와 동일 출처 (정적 입지변수)

N_NEIGHBORS = 5
MIN_TRADES_DEFAULT = 10  # 커버리지 표를 본 뒤 조정 가능
EARTH_RADIUS_KM = 6371.0

SLOPE_RADII = [100, 150, 200, 300]
SLOPE_STATS = ["mean", "median", "stdev", "min", "max", "range"]
TRANSIT_COLS = ["straight_dist_m", "walk_dist_m", "walk_time_min", "avg_daily_ridership"]
STATION_ZONE_ORDER = {"초역세권": 1, "역세권": 2, "준역세권": 3, "비역세권": 4}

# gwr_feature_selection.py 중첩 K-fold LASSO 결과 (data/csv/gwr_lasso_coefs_{MDD,CV}.csv) —
# 계수가 0이 아니었던 변수만 채택. 반경은 타겟별로 최적 반경이 달라 각각 다름
# (MDD: 100m 채택, CV: 150m 채택 — 두 스크리닝 결과에서 그대로 가져옴)
LASSO_SELECTED_MDD = ["slope_max_100m", "avg_daily_ridership", "straight_dist_m",
                       "slope_range_100m", "station_zone_ord"]
LASSO_SELECTED_CV = ["slope_median_150m", "station_zone_ord", "straight_dist_m"]
# walk_dist_m은 제외 — straight_dist_m과 상관관계 1.0(완전공선성)으로 GWR 국지회귀에서
# 두 계수가 서로 상쇄되며 폭주(+-16 수준)하는 걸 확인해 제거


# ------------------------------------------------------------------
# 1. 단지 테이블 + 리스크 타겟(MDD/CV) 생성
# ------------------------------------------------------------------
def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["year_month"] = df["contract_date"].dt.to_period("M")
    return df


def trade_count_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """최소거래건수 임계값별 단지수/거래 커버리지 트레이드오프 표"""
    counts = df.groupby("complex_id").size()
    total_c, total_t = len(counts), len(df)
    rows = []
    for th in [1, 3, 5, 10, 15, 20, 30]:
        kc = (counts >= th).sum()
        kt = counts[counts >= th].sum()
        rows.append({
            "min_trades": th,
            "n_complex": kc,
            "complex_pct": round(kc / total_c * 100, 1),
            "n_trades_covered": int(kt),
            "trade_pct": round(kt / total_t * 100, 1),
        })
    return pd.DataFrame(rows)


def compute_mdd(price_series: pd.Series) -> float:
    if len(price_series) < 2:
        return np.nan
    cummax = price_series.cummax()
    drawdown = (price_series - cummax) / cummax
    return drawdown.min() * -1


def build_complex_table(df: pd.DataFrame, min_trades: int) -> pd.DataFrame:
    """단지별 정적 스냅샷 1행: 좌표, 물건특성 평균, MDD/CV 타겟"""
    counts = df.groupby("complex_id").size()
    valid_ids = counts[counts >= min_trades].index

    rows = []
    for cid, g in df[df["complex_id"].isin(valid_ids)].groupby("complex_id"):
        monthly = g.groupby("year_month")["price_per_m2"].mean().sort_index()
        cv = g["price_per_m2"].std() / g["price_per_m2"].mean()
        mdd = compute_mdd(monthly)
        rows.append({
            "complex_id": cid,
            "aptNm": g["aptNm"].iloc[0],
            "umdNm": g["umdNm"].iloc[0],
            "jibun": g["jibun"].iloc[0],
            "lat": g["lat"].iloc[0],
            "lon": g["lon"].iloc[0],
            "n_trades": len(g),
            "mean_price_per_m2": g["price_per_m2"].mean(),
            "excluUseAr": g["excluUseAr"].mean(),
            "floor": g["floor"].mean(),
            "building_age": (2025 - g["buildYear"]).mean(),
            "MDD": mdd,
            "CV": cv,
        })
    table = pd.DataFrame(rows).dropna(subset=["MDD"]).reset_index(drop=True)
    print(f"[단지 테이블] min_trades={min_trades} -> {len(table)}개 단지 "
          f"(MDD 계산 불가 {len(valid_ids) - len(table)}개 제외)")
    return table


# ------------------------------------------------------------------
# 2. 이웃단지 매핑 (BallTree, haversine) + 공간 동적 특성
# ------------------------------------------------------------------
def add_neighbor_features(table: pd.DataFrame, k: int = N_NEIGHBORS) -> pd.DataFrame:
    coords_rad = np.radians(table[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    k_query = min(k + 1, len(table))  # 자기 자신 포함해서 조회 후 제외
    dist, idx = tree.query(coords_rad, k=k_query)

    neighbor_price = []
    neighbor_dist_km = []
    for i in range(len(table)):
        neighbor_idx = [j for j in idx[i] if j != i][:k]
        neighbor_d = [d for j, d in zip(idx[i], dist[i]) if j != i][:k]
        neighbor_price.append(table.iloc[neighbor_idx]["mean_price_per_m2"].mean())
        neighbor_dist_km.append(np.mean(neighbor_d) * EARTH_RADIUS_KM)

    table = table.copy()
    table["neighbor_price_level"] = neighbor_price
    table["neighbor_dist_km"] = neighbor_dist_km
    return table


def add_neighbor_momentum(table: pd.DataFrame, df: pd.DataFrame, k: int = N_NEIGHBORS) -> pd.DataFrame:
    """이웃단지들의 최근 3개월 대비 이전 3개월 가격 모멘텀(%) — 전체 데이터 기준 '현재 시점' 스냅샷"""
    last_month = df["year_month"].max()
    recent_3m = df[df["year_month"] > last_month - 3]
    prior_3m = df[(df["year_month"] <= last_month - 3) & (df["year_month"] > last_month - 6)]

    recent_price = recent_3m.groupby("complex_id")["price_per_m2"].mean()
    prior_price = prior_3m.groupby("complex_id")["price_per_m2"].mean()

    coords_rad = np.radians(table[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    k_query = min(k + 1, len(table))
    _, idx = tree.query(coords_rad, k=k_query)

    momentum = []
    for i, cid in enumerate(table["complex_id"]):
        neighbor_ids = [table.iloc[j]["complex_id"] for j in idx[i] if j != i][:k]
        r = recent_price.reindex(neighbor_ids).mean()
        p = prior_price.reindex(neighbor_ids).mean()
        momentum.append((r / p - 1) * 100 if p and not np.isnan(p) and p != 0 else np.nan)

    table = table.copy()
    table["neighbor_momentum_3m"] = momentum
    print(f"[모멘텀] 기준: 최근 3개월({(last_month-2).strftime('%Y-%m')}~{last_month.strftime('%Y-%m')}) "
          f"vs 이전 3개월({(last_month-5).strftime('%Y-%m')}~{(last_month-3).strftime('%Y-%m')})")
    return table


# ------------------------------------------------------------------
# 3. 정적 입지변수 (data/csv/final.csv) — gwr_feature_selection.py와 동일 로직
#    (두 파일이 서로를 import하면 순환참조가 생기므로 여기 그대로 복제)
# ------------------------------------------------------------------
def load_static_features() -> pd.DataFrame:
    df = pd.read_csv(FINAL_CSV)

    static_cols = [f"slope_{stat}_{r}m" for r in SLOPE_RADII for stat in SLOPE_STATS]
    static_cols += ["school_pc1"] + TRANSIT_COLS + ["station_zone"]

    table = df.groupby("complex_id")[static_cols].first().reset_index()
    table["station_zone_ord"] = table["station_zone"].map(STATION_ZONE_ORDER)
    table = table.drop(columns=["station_zone"])
    return table


# ------------------------------------------------------------------
# 4. GWR 실행
# ------------------------------------------------------------------
def run_gwr(table: pd.DataFrame, feature_cols: list[str], target_col: str, label: str):
    sub = table.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"\n{'='*70}\n[GWR: {label}] 표본 n={len(sub)}, 종속변수={target_col}, "
          f"독립변수={feature_cols}")

    if len(sub) < 30:
        print("  -> 표본이 너무 적어(<30) GWR 신뢰도가 낮음. 스킵.")
        return None

    coords = list(zip(sub["lon"], sub["lat"]))
    y = sub[[target_col]].values
    X_raw = sub[feature_cols].values
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

    selector = Sel_BW(coords, y, X)
    bw = selector.search()
    model = GWR(coords, y, X, bw)
    results = model.fit()

    n, k = len(sub), len(feature_cols)
    adj_r2 = 1 - (1 - results.R2) * (n - 1) / (n - k - 1)

    print(f"  대역폭(bandwidth) = {bw:.1f} (표본 {len(sub)}개 중 이웃 수, "
          f"전체표본에 근접할수록 국지적 이질성이 옅다는 뜻)")
    print(f"  Global R2 = {results.R2:.3f}  (조정R2 = {adj_r2:.3f}, 변수 {k}개 기준)  "
          f"AICc = {results.aicc:.2f}")

    tvals = results.tvalues
    var_names = ["intercept"] + feature_cols
    print("\n  변수별 유의비율 (|t| > 1.96 기준):")
    for i, name in enumerate(var_names):
        sig_rate = (np.abs(tvals[:, i]) > 1.96).mean()
        mean_coef = results.params[:, i].mean()
        print(f"    {name:25s} 유의비율={sig_rate:6.1%}  평균계수={mean_coef:+.4f}")

    return {"results": results, "sub": sub, "r2": results.R2, "adj_r2": adj_r2,
            "aicc": results.aicc, "n": n, "k": k, "bw": bw}


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    df = load_master()

    print("=== 최소거래건수 임계값별 커버리지 ===")
    print(trade_count_coverage(df).to_string(index=False))

    table = build_complex_table(df, MIN_TRADES_DEFAULT)
    table = add_neighbor_features(table)
    table = add_neighbor_momentum(table, df)

    table.to_csv("data/csv/gwr_complex_table.csv", index=False, encoding="utf-8-sig")
    print(f"\n[저장] 단지 테이블 -> data/csv/gwr_complex_table.csv ({len(table)}행)")

    baseline_features = ["excluUseAr", "building_age", "floor",
                          "neighbor_price_level", "neighbor_momentum_3m"]

    static = load_static_features()
    table = table.merge(static, on="complex_id", how="left")

    comparisons = [
        ("MDD", baseline_features, baseline_features + LASSO_SELECTED_MDD),
        ("CV", baseline_features, baseline_features + LASSO_SELECTED_CV),
    ]

    summary = []
    for target_col, base_cols, ext_cols in comparisons:
        base = run_gwr(table, base_cols, target_col, f"baseline ({target_col})")
        ext = run_gwr(table, ext_cols, target_col, f"baseline+입지특성 ({target_col})")
        if base is None or ext is None:
            continue
        delta_aicc = ext["aicc"] - base["aicc"]
        if delta_aicc <= -10:
            verdict = "강한 개선"
        elif delta_aicc <= -4:
            verdict = "꽤 확실한 개선"
        elif delta_aicc <= -2:
            verdict = "약한 개선"
        else:
            verdict = "차이 없음/악화"
        summary.append({
            "target": target_col,
            "baseline_R2": base["r2"], "extended_R2": ext["r2"],
            "delta_R2": ext["r2"] - base["r2"],
            "baseline_adjR2": base["adj_r2"], "extended_adjR2": ext["adj_r2"],
            "delta_adjR2": ext["adj_r2"] - base["adj_r2"],
            "baseline_AICc": base["aicc"], "extended_AICc": ext["aicc"],
            "delta_AICc": delta_aicc, "AICc_판정": verdict,
        })

    summary_df = pd.DataFrame(summary)
    print(f"\n{'='*70}\n[베이스라인 vs 입지특성 추가 비교 — 가설: 입지특성 추가 시 설명력 상승]")
    print(f"(AICc는 낮을수록 좋은 모델. delta_AICc = 확장 - 베이스라인이 음수일수록 확장판이 더 나음)")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("data/csv/gwr_baseline_vs_extended.csv", index=False, encoding="utf-8-sig")
    print("[저장] -> data/csv/gwr_baseline_vs_extended.csv")


if __name__ == "__main__":
    main()
