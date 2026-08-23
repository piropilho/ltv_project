"""
유동성(trade_activity_ratio) 비선형 모델 체크
================================================
liquidity_feature_selection.py의 LassoCV(선형)는 정적+시변평균 조합에서
중첩 K-fold R2=0.159까지 나왔다. LASSO는 선형관계만 잡기 때문에, "역세권 500m 이내는
급격히 유동성이 오르고 그 밖은 평평하다"류의 임계치/비선형 패턴이 있다면 놓칠 수 있다.
같은 피처·같은 타겟·같은 반복 K-fold 구조로 XGBoost를 돌려 비선형 모델이 유의미하게
더 잡아내는 신호가 있는지 확인한다 (같은 조건이어야 LASSO 결과와 공정하게 비교 가능).

n=316으로 표본이 작아 트리 개수/깊이를 train_model.py의 XGB_PARAMS보다 보수적으로
줄였다 (과적합 방지).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트(liquidity_feature_selection.py 위치) 참조용
from liquidity_feature_selection import (
    FINAL_CSV, TIME_VARYING_COLS, SLOPE_RADII, SLOPE_STATS,
    TRANSIT_COLS, N_SPLITS, N_REPEATS, build_static_table,
)

BEST_RADIUS = 300  # combined LASSO에서 채택된 반경 그대로 재사용

XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=42,
    objective="reg:squarederror",
)


def build_combined_table() -> pd.DataFrame:
    static_table = build_static_table()
    df = pd.read_csv(FINAL_CSV)
    time_avg = df.groupby("complex_id")[TIME_VARYING_COLS].mean().reset_index()
    time_avg.columns = ["complex_id"] + [f"{c}_avg" for c in TIME_VARYING_COLS]
    return static_table.merge(time_avg, on="complex_id", how="inner")


def repeated_kfold_r2_xgb(X: pd.DataFrame, y: pd.Series) -> float:
    scores = []
    for seed in range(N_REPEATS):
        outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, te_idx in outer.split(X):
            model = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = model.predict(X.iloc[te_idx])
            ss_res = ((y.iloc[te_idx] - pred) ** 2).sum()
            ss_tot = ((y.iloc[te_idx] - y.iloc[te_idx].mean()) ** 2).sum()
            scores.append(1 - ss_res / ss_tot)
    return float(np.mean(scores))


def main():
    table = build_combined_table()
    target_col = "trade_activity_ratio"

    slope_cols = [f"slope_{stat}_{BEST_RADIUS}m" for stat in SLOPE_STATS]
    time_avg_cols = [f"{c}_avg" for c in TIME_VARYING_COLS]
    feature_cols = slope_cols + ["school_pc1"] + TRANSIT_COLS + ["station_zone_ord", "unit_cnt"] + time_avg_cols

    sub = table.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X, y = sub[feature_cols], sub[target_col]

    print(f"[데이터] n={len(sub)}, 피처 {len(feature_cols)}개 (반경 {BEST_RADIUS}m, combined LASSO와 동일)")

    r2 = repeated_kfold_r2_xgb(X, y)
    print(f"\n[XGBoost] 반복 K-fold out-of-sample R2={r2:.4f}  (참고: 같은 피처셋 LassoCV 중첩 R2=0.159)")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[전체데이터 적합] 변수중요도(gain 기반) 상위 10개:")
    for name, imp in importance.head(10).items():
        print(f"    {name:28s} {imp:.4f}")

    pd.DataFrame([{"model": "xgboost", "repeated_kfold_r2": r2, "n": len(sub)}]).to_csv(
        "data/csv/liquidity_xgboost_result_summary.csv", index=False, encoding="utf-8-sig")
    importance.to_csv("data/csv/liquidity_xgboost_importance.csv", encoding="utf-8-sig")
    print("\n[저장] data/csv/liquidity_xgboost_result_summary.csv, data/csv/liquidity_xgboost_importance.csv")


if __name__ == "__main__":
    main()
