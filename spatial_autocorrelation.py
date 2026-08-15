"""
공간 자기상관 감쇠 분석 — 이웃단지 집계(k)를 데이터 기반으로 정하기 위한 진단 스크립트
=================================================================================
입력: data/master_19_25_cleaning.csv

방법:
1. 단지 쌍(i, j) 사이의 물리적 거리(km)와 가격 유사도의 관계를 본다.
2. 거리를 구간(bin)으로 나누고, 각 구간에 속한 단지쌍들의 (price_i, price_j) 상관계수를
   계산한다 (경험적 상관로그램). 동시에 준분산(semivariance)도 계산해 교차검증한다
   (거리가 멀수록 상관은 줄고 준분산은 늘어야 서로 같은 이야기를 하는 것).
3. GWR/XGBoost에서 실제로 쓸 법한 k(3/5/8/10/15/20)별로, 그 k가 평균적으로 어느 반경까지
   이웃을 끌어오는지, 그 반경에서의 상관관계가 얼마인지 표로 정리한다.
   -> "k=5가 통계적으로 말이 되는 선택인지"를 직접 판단할 수 있는 근거표.

주의: 단지별 대표가격은 전체기간 평균 ㎡당가격 1개 값만 쓴다 (월별 패널로 하면 거래가 희박한
단지 쌍이 너무 많아 상관계수 자체가 노이즈에 묻힘). 따라서 이 분석은 "가격 수준의 공간적
유사성"을 보는 것이지 "가격 변화(모멘텀)의 공간적 유사성"을 보는 게 아니라는 점을 감안할 것.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

MASTER_CSV = "data/master_19_25_cleaning.csv"
MIN_TRADES = 5          # 평균가가 너무 불안정한 단지는 제외 (조정 가능)
BIN_WIDTH_KM = 0.3
MAX_DIST_KM = 6.0
MIN_PAIRS_PER_BIN = 30  # 이보다 적은 구간은 상관계수가 불안정하므로 제외
K_CANDIDATES = [3, 5, 8, 10, 15, 20]
EARTH_RADIUS_KM = 6371.0


def load_complex_table() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    counts = df.groupby("complex_id").size()
    valid = counts[counts >= MIN_TRADES].index
    table = (
        df[df["complex_id"].isin(valid)]
        .groupby("complex_id")
        .agg(lat=("lat", "first"), lon=("lon", "first"),
             mean_price_per_m2=("price_per_m2", "mean"),
             n_trades=("price_per_m2", "size"))
        .reset_index()
    )
    print(f"[단지 테이블] min_trades={MIN_TRADES} 기준 {len(table)}개 단지")
    return table


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_pairwise(table: pd.DataFrame) -> pd.DataFrame:
    lat, lon, price = table["lat"].values, table["lon"].values, table["mean_price_per_m2"].values
    i_idx, j_idx = np.triu_indices(len(table), k=1)
    dist = haversine_km(lat[i_idx], lon[i_idx], lat[j_idx], lon[j_idx])
    pairs = pd.DataFrame({"dist_km": dist, "price_i": price[i_idx], "price_j": price[j_idx]})
    print(f"[단지쌍] 총 {len(pairs)}쌍, 거리범위 {pairs.dist_km.min():.2f}~{pairs.dist_km.max():.2f}km")
    return pairs


def correlogram(pairs: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0, MAX_DIST_KM + BIN_WIDTH_KM, BIN_WIDTH_KM)
    pairs = pairs.copy()
    pairs["bin"] = pd.cut(pairs["dist_km"], bins)

    rows = []
    for b, g in pairs.groupby("bin", observed=True):
        if len(g) < MIN_PAIRS_PER_BIN:
            continue
        corr = np.corrcoef(g["price_i"], g["price_j"])[0, 1]
        semivar = 0.5 * np.mean((g["price_i"] - g["price_j"]) ** 2)
        rows.append({"dist_bin_mid": b.mid, "n_pairs": len(g),
                     "correlation": corr, "semivariance": semivar})
    return pd.DataFrame(rows)


def k_to_radius_table(table: pd.DataFrame, corr_result: pd.DataFrame) -> pd.DataFrame:
    """k(이웃 수)별로 평균 반경과, 그 반경에서의 상관관계를 매핑"""
    coords_rad = np.radians(table[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    rows = []
    for k in K_CANDIDATES:
        dist, _ = tree.query(coords_rad, k=k + 1)  # 자기 자신 포함 조회
        avg_radius_km = dist[:, 1:].mean() * EARTH_RADIUS_KM

        nearest = (corr_result["dist_bin_mid"] - avg_radius_km).abs().idxmin()
        corr_at_radius = corr_result.loc[nearest, "correlation"]

        rows.append({"k": k, "avg_radius_km": round(avg_radius_km, 2),
                     "corr_at_that_radius": round(float(corr_at_radius), 3)})
    return pd.DataFrame(rows)


def main():
    table = load_complex_table()
    pairs = build_pairwise(table)
    corr_result = correlogram(pairs)

    print("\n=== 거리 구간별 가격 상관관계 / 준분산 ===")
    print(corr_result.to_string(index=False))
    corr_result.to_csv("data/spatial_autocorrelation.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] data/spatial_autocorrelation.csv")

    print("\n=== k(이웃 수)별 평균 반경 및 해당 반경에서의 상관관계 ===")
    k_table = k_to_radius_table(table, corr_result)
    print(k_table.to_string(index=False))
    k_table.to_csv("data/spatial_autocorrelation_k_table.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] data/spatial_autocorrelation_k_table.csv")


if __name__ == "__main__":
    main()
