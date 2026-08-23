"""
GWR 트랙 시변(time-varying) 입지변수 변수선택 — 단지 고정효과 제거 후 가격편차 타겟
=====================================================================================
정적 변수(gwr_feature_selection.py, 경사도/학군/교통)는 MDD/CV를 타겟으로 "단지 1개=1행"
K-fold를 썼다. 하지만 상권지수/인구/건물연식처럼 분기마다 값이 바뀌는 변수는 그 방식을
쓸 수 없다 — 단지 고정효과로 100% 흡수되는 정적 변수와 달리, 이 변수들은 단지 내부에서도
분산이 있기 때문이다. 그래서 여기서는 commercial_index_lasso19가 원래 검증됐던 방식
("단지 고정효과 제거 후 남는 가격편차" 타겟, 단지×분기 패널 그대로 사용)을 그대로 재사용해
후보를 확장한다.

관측 단위: 단지×분기 (실거래가 있어 avg_price_per_m2가 채워진 행만)
타겟:      avg_price_per_m2 - 단지평균가  (단지 고정효과 제거 후 가격편차)
후보 변수: commercial_index_lasso19, population, living_population, pop_growth_mom,
           building_age (재계산: yyyymm 연도 - buildYear. 원본 컬럼은 "2026년 기준"으로
           고정되어 있던 버그를 여기서 바로잡음)
검증:      GroupKFold(단지 단위 완전분리) — 같은 단지의 여러 분기가 train/test에 걸쳐
           있으면 "이 단지는 원래 비싸다"는 정보가 새어나가 성능이 낙관적으로 왜곡된다.
           일반 K-fold는 부적절 (commercial_index_lasso19 개발 때도 이 문제로 GroupKFold
           채택, 이전에는 일반 K-fold에서 alpha가 폴드마다 불안정하게 흔들렸음).

정적 변수(경사도/학군/교통)는 여기 포함하지 않는다 — demean(단지 고정효과 제거) 순간
단지 내부 분산이 0이라 계수 추정 자체가 불가능해지기 때문 (gwr_feature_selection.py에서
이미 진단한 구조적 불일치).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINAL_CSV = "data/csv/final.csv"

CANDIDATE_COLS = ["commercial_index_lasso19", "population", "living_population",
                   "pop_growth_mom", "building_age"]

N_SPLITS = 5
N_REPEATS = 20
LASSO_MAX_ITER = 100000
MIN_QUARTERS_PER_COMPLEX = 2  # 1분기만 있으면 demean 후 값이 항상 0 -> 정보량 없음


# ------------------------------------------------------------------
# 1. 패널 로드 + 타겟(가격편차) 구성
# ------------------------------------------------------------------
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(FINAL_CSV)
    df["building_age"] = df["yyyymm"] // 100 - df["buildYear"]

    df = df.dropna(subset=["avg_price_per_m2"]).copy()

    n_quarters = df.groupby("complex_id")["yyyymm"].transform("size")
    df = df[n_quarters >= MIN_QUARTERS_PER_COMPLEX].copy()

    df["price_deviation"] = df["avg_price_per_m2"] - df.groupby("complex_id")["avg_price_per_m2"].transform("mean")

    print(f"[패널] 실거래 있는 단지×분기 {len(df)}행, {df['complex_id'].nunique()}개 단지 "
          f"(분기가 {MIN_QUARTERS_PER_COMPLEX}개 미만인 단지는 demean 시 값이 항상 0이 되어 자동 제외)")
    return df


# ------------------------------------------------------------------
# 2. 후보 변수 간 상관관계 점검 (다중공선성 사전 확인)
# ------------------------------------------------------------------
def check_collinearity(df: pd.DataFrame):
    corr = df[CANDIDATE_COLS].corr()
    print("\n[후보 변수 상관행렬]")
    print(corr.round(3))


# ------------------------------------------------------------------
# 3. 중첩 GroupKFold — alpha 선택과 평가를 분리해 편향 없이 R2 산출
# ------------------------------------------------------------------
def nested_groupkfold_r2(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    scores = []
    rng = np.random.default_rng(0)
    n = len(X)
    for rep in range(N_REPEATS):
        perm = rng.permutation(n)
        Xp = X.iloc[perm].reset_index(drop=True)
        yp = y.iloc[perm].reset_index(drop=True)
        gp = groups.iloc[perm].reset_index(drop=True)

        outer = GroupKFold(n_splits=N_SPLITS)
        for tr_idx, te_idx in outer.split(Xp, yp, gp):
            inner_gp = gp.iloc[tr_idx]
            inner_cv = list(GroupKFold(n_splits=N_SPLITS).split(Xp.iloc[tr_idx], yp.iloc[tr_idx], inner_gp))
            model = make_pipeline(
                StandardScaler(),
                LassoCV(cv=inner_cv, max_iter=LASSO_MAX_ITER),
            )
            model.fit(Xp.iloc[tr_idx], yp.iloc[tr_idx])
            scores.append(model.score(Xp.iloc[te_idx], yp.iloc[te_idx]))
    return float(np.mean(scores))


# ------------------------------------------------------------------
# 4. 최종 변수선택 — 전체 데이터로 GroupKFold LassoCV 1회 적합
# ------------------------------------------------------------------
def fit_final_lasso(df: pd.DataFrame, feature_cols: list, target_col: str):
    sub = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    X, y, groups = sub[feature_cols], sub[target_col], sub["complex_id"]

    cv = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LassoCV(cv=cv, max_iter=LASSO_MAX_ITER, alphas=200)
    model.fit(X_scaled, y)

    coefs = pd.Series(model.coef_, index=feature_cols).sort_values(key=np.abs, ascending=False)
    selected = coefs[coefs != 0]

    print(f"\n  alpha={model.alpha_:.4f}  R2(전체데이터 적합)={model.score(X_scaled, y):.3f}  n={len(sub)}")
    print(f"  선택된 변수 {len(selected)}/{len(feature_cols)}개:")
    for name, coef in selected.items():
        print(f"    {name:28s} {coef:+.4f}")
    return model, coefs


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    df = load_panel()
    check_collinearity(df)

    target_col = "price_deviation"
    sub = df.dropna(subset=CANDIDATE_COLS + [target_col])
    X, y, groups = sub[CANDIDATE_COLS], sub[target_col], sub["complex_id"]

    print(f"\n{'='*70}\n[타겟: {target_col}] 중첩 GroupKFold out-of-sample R2\n{'='*70}")
    r2 = nested_groupkfold_r2(X, y, groups)
    print(f"  중첩 GroupKFold R2={r2:.4f} (n={len(sub)}, 단지수={groups.nunique()})")

    print(f"\n[타겟: {target_col}] 최종 LASSO (전체 후보 변수 {len(CANDIDATE_COLS)}개)")
    _, coefs = fit_final_lasso(df, CANDIDATE_COLS, target_col)

    result_summary = pd.DataFrame([{"target": target_col, "nested_groupkfold_r2": r2, "n": len(sub)}])
    result_summary.to_csv("data/csv/panel_lasso_result_summary.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv("data/csv/panel_lasso_coefs.csv", encoding="utf-8-sig")
    df.to_csv("data/csv/panel_features_table.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] data/csv/panel_lasso_result_summary.csv, data/csv/panel_lasso_coefs.csv, "
          "data/csv/panel_features_table.csv")


if __name__ == "__main__":
    main()
