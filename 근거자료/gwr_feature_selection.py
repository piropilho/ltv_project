"""
GWR 트랙 정적 입지변수 변수선택 — 레벨 기반 LASSO (commercial_index_lasso19와는 다른 방식)
=========================================================================================
배경: commercial_index_lasso19는 "단지 고정효과(단지별 평균가) 제거한 가격편차"를 타겟으로
      LASSO를 돌렸다. 이 방식은 상권/인구처럼 분기마다 값이 바뀌는 변수엔 맞지만, 경사도/
      학군/교통처럼 시간에 안 변하는 정적 변수엔 원리적으로 적용 불가하다 — 단지 고정효과를
      제거하면 시간불변 변수의 분산이 0이 되어버려 계수 추정 자체가 불가능해지기 때문.
      GWR이 설명하려는 것도 정확히 그 "단지 고정효과"(공간적 이질성) 쪽이므로, 이 스크립트는
      타겟을 가격편차가 아니라 단지별 리스크 레벨(MDD/CV, gwr_analysis.py와 동일 정의)로
      바꿔서 별도로 변수선택한다.

관측 단위: 단지 1개 = 1행 (정적 변수라 패널 유지 의미 없음, 28분기 반복은 단순 복제일 뿐)
타겟: MDD(주력), CV(보조) — gwr_analysis.py의 build_complex_table() 그대로 재사용
후보 변수: 경사도 1개 반경(4개 중 중첩 K-fold로 비교해 선택) + school_pc1 + 교통 5개
          building_age는 스크리닝 대상이 아님 — gwr_analysis.py에 이미 강제 포함되는 통제변수
검증: 일반 K-fold (GroupKFold 아님 — 단지당 1행이라 그룹 누수 자체가 발생하지 않음).
      alpha 선택 자체도 데이터에 의존하므로, 반경 비교 단계는 중첩(nested) K-fold로 평가해
      상권지수 개발 때 겪었던 "폴드 나누는 방식에 따라 alpha가 흔들리는" 문제를 방지.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트(gwr_analysis.py 위치) 참조용
from gwr_analysis import load_master, build_complex_table, MIN_TRADES_DEFAULT

FINAL_CSV = "data/csv/final.csv"

SLOPE_RADII = [100, 150, 200, 300]
SLOPE_STATS = ["mean", "median", "stdev", "min", "max", "range"]

TRANSIT_COLS = ["straight_dist_m", "walk_dist_m", "walk_time_min", "avg_daily_ridership"]
STATION_ZONE_ORDER = {"초역세권": 1, "역세권": 2, "준역세권": 3, "비역세권": 4}

N_SPLITS = 5          # K-fold 개수
N_REPEATS = 20         # 시드를 바꿔가며 반복 (alpha 불안정성 방지)
LASSO_MAX_ITER = 100000


# ------------------------------------------------------------------
# 1. 정적 입지변수 테이블 (단지 1개 = 1행)
# ------------------------------------------------------------------
def load_static_features() -> pd.DataFrame:
    df = pd.read_csv(FINAL_CSV)

    static_cols = [f"slope_{stat}_{r}m" for r in SLOPE_RADII for stat in SLOPE_STATS]
    static_cols += ["school_pc1"] + TRANSIT_COLS + ["station_zone"]

    table = df.groupby("complex_id")[static_cols].first().reset_index()
    table["station_zone_ord"] = table["station_zone"].map(STATION_ZONE_ORDER)
    table = table.drop(columns=["station_zone"])

    print(f"[정적 입지변수] {len(table)}개 단지 (complex_id 기준 — 휘경주공1/2단지처럼 "
          f"apt_name으로만 구분되는 케이스는 complex_id가 같아 1개로 합쳐짐)")
    return table


# ------------------------------------------------------------------
# 2. 리스크 타겟 (MDD/CV) — gwr_analysis.py 재사용
# ------------------------------------------------------------------
def load_risk_targets() -> pd.DataFrame:
    df = load_master()
    table = build_complex_table(df, MIN_TRADES_DEFAULT)
    return table[["complex_id", "MDD", "CV"]]


# ------------------------------------------------------------------
# 3. 경사도 반경 비교 — 중첩 K-fold로 편향 없이 비교
# ------------------------------------------------------------------
def nested_cv_r2(X: pd.DataFrame, y: pd.Series) -> float:
    scores = []
    for seed in range(N_REPEATS):
        outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, te_idx in outer.split(X):
            inner = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            model = make_pipeline(
                StandardScaler(),
                LassoCV(cv=inner, random_state=seed, max_iter=LASSO_MAX_ITER),
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            scores.append(model.score(X.iloc[te_idx], y.iloc[te_idx]))
    return float(np.mean(scores))


def select_slope_radius(table: pd.DataFrame, other_features: list, target_col: str) -> tuple[int, pd.DataFrame]:
    results = []
    for radius in SLOPE_RADII:
        slope_cols = [f"slope_{stat}_{radius}m" for stat in SLOPE_STATS]
        feature_cols = slope_cols + other_features
        sub = table.dropna(subset=feature_cols + [target_col])
        r2 = nested_cv_r2(sub[feature_cols], sub[target_col])
        results.append({"radius_m": radius, "nested_cv_r2": r2, "n": len(sub)})
        print(f"  반경 {radius}m: 중첩 K-fold R2={r2:.4f} (n={len(sub)})")

    result_df = pd.DataFrame(results)
    best_radius = int(result_df.loc[result_df["nested_cv_r2"].idxmax(), "radius_m"])
    return best_radius, result_df


# ------------------------------------------------------------------
# 4. 최종 변수선택 — 선택된 반경으로 전체 데이터 LassoCV 1회 적합
# ------------------------------------------------------------------
def fit_final_lasso(table: pd.DataFrame, feature_cols: list, target_col: str):
    sub = table.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X, y = sub[feature_cols], sub[target_col]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(cv=cv, random_state=0, max_iter=LASSO_MAX_ITER, alphas=200)
    model.fit(X_scaled, y)

    coefs = pd.Series(model.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    selected = coefs[coefs != 0]

    print(f"\n  alpha={model.alpha_:.4f}  R2(전체데이터 적합)={model.score(X_scaled, y):.3f}  n={len(sub)}")
    print(f"  선택된 변수 {len(selected)}/{len(feature_cols)}개:")
    for name, coef in selected.items():
        flip_note = " (school_pc1: 부호 반전 필요 — 값이 클수록 리스크는 낮은데 계수 부호는 원본 기준)" \
            if name == "school_pc1" else ""
        print(f"    {name:28s} {coef:+.4f}{flip_note}")
    return model, coefs


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    static = load_static_features()
    risk = load_risk_targets()
    table = static.merge(risk, on="complex_id", how="inner")
    print(f"[병합] 정적변수 x 리스크타겟 -> {len(table)}개 단지\n")

    other_features = ["school_pc1", "straight_dist_m", "walk_dist_m", "walk_time_min",
                       "avg_daily_ridership", "station_zone_ord"]

    for target_col in ["MDD", "CV"]:
        print(f"{'='*70}\n[타겟: {target_col}] 경사도 반경별 중첩 K-fold 비교\n{'='*70}")
        best_radius, radius_results = select_slope_radius(table, other_features, target_col)
        radius_results.to_csv(f"data/csv/gwr_slope_radius_comparison_{target_col}.csv",
                               index=False, encoding="utf-8-sig")
        print(f"  -> 채택 반경: {best_radius}m\n")

        slope_cols = [f"slope_{stat}_{best_radius}m" for stat in SLOPE_STATS]
        feature_cols = slope_cols + other_features
        print(f"[타겟: {target_col}] 최종 LASSO (반경 {best_radius}m + 학군 + 교통)")
        _, coefs = fit_final_lasso(table, feature_cols, target_col)
        coefs.to_csv(f"data/csv/gwr_lasso_coefs_{target_col}.csv", encoding="utf-8-sig")
        print()

    table.to_csv("data/csv/gwr_static_features_table.csv", index=False, encoding="utf-8-sig")
    print("[저장] data/csv/gwr_static_features_table.csv, "
          "data/csv/gwr_slope_radius_comparison_{MDD,CV}.csv, data/csv/gwr_lasso_coefs_{MDD,CV}.csv")


if __name__ == "__main__":
    main()
