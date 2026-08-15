"""
트랙 A: GWR(지리적 가중회귀) — "왜 지금 이 단지가 위험한가"를 설명하는 정적 스냅샷 분석
=======================================================================================
입력: data/master_19_25_cleaning.csv (7개년, 15,365건, 316개 단지, 좌표 포함) 단독 사용
      — 학군/지하철/상권 등 입지변수는 아직 결합하지 않음 (실거래가 데이터만으로 베이스라인 산출)

설계 원칙 (기획안 반영):
- 관측 단위: 좌표점(단지) 1개 = 1행, 정적 스냅샷 (거래 단위 아님)
- 종속변수: MDD(최대낙폭)를 주력 지표로, CV(변동계수)를 보조지표로 병행 산출
  (담보가치 리스크는 "얼마나 흩어지는가"보다 "고점 대비 얼마나 빠지는가"가 은행 관점에 더 부합)
- 최소거래건수 임계값은 고정하지 않고, 커버리지 트레이드오프 표를 먼저 보여준 뒤 결정
- 검증은 없음(전체표본 회귀) — 국지적 유의성(t-value)으로 판단
- 좌표는 mgwr 규약에 따라 (lon, lat) 순서로 입력, 독립변수는 표준화
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

MASTER_CSV = "data/master_19_25_cleaning.csv"

N_NEIGHBORS = 5
MIN_TRADES_DEFAULT = 10  # 커버리지 표를 본 뒤 조정 가능
EARTH_RADIUS_KM = 6371.0


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
# 3. GWR 실행
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

    print(f"  대역폭(bandwidth) = {bw:.1f} (표본 {len(sub)}개 중 이웃 수, "
          f"전체표본에 근접할수록 국지적 이질성이 옅다는 뜻)")
    print(f"  Global R2 = {results.R2:.3f}")

    tvals = results.tvalues
    var_names = ["intercept"] + feature_cols
    print("\n  변수별 유의비율 (|t| > 1.96 기준):")
    for i, name in enumerate(var_names):
        sig_rate = (np.abs(tvals[:, i]) > 1.96).mean()
        mean_coef = results.params[:, i].mean()
        print(f"    {name:25s} 유의비율={sig_rate:6.1%}  평균계수={mean_coef:+.4f}")

    return results, sub


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

    table.to_csv("data/gwr_complex_table.csv", index=False, encoding="utf-8-sig")
    print(f"\n[저장] 단지 테이블 -> data/gwr_complex_table.csv ({len(table)}행)")

    baseline_features = ["excluUseAr", "building_age", "floor",
                          "neighbor_price_level", "neighbor_momentum_3m"]

    run_gwr(table, baseline_features, "MDD", "baseline (MDD, 주력지표)")
    run_gwr(table, baseline_features, "CV", "baseline (CV, 보조지표)")


if __name__ == "__main__":
    main()
