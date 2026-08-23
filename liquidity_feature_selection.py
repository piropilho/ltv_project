"""
입지특성 → 유동성(liquidity) 리스크 설명력 검증
=================================================
배경: 입지특성(경사도/학군/교통 등 정적, 상권/인구/건물연식 등 시변) 모두 가격 변동성
      (MDD/CV, 가격편차)은 설명하지 못했다 (gwr_feature_selection.py, panel_feature_selection.py
      결과 R2 0.03~0.05 수준). 프로젝트가 정의한 리스크는 "변동성 + 유동성" 두 축인데,
      지금까지는 변동성 쪽만 테스트했다. 역세권/상권처럼 입지특성은 가격이 얼마나
      흔들리는가보다 "이 단지가 얼마나 자주/쉽게 거래되는가(환금성)"와 더 직접적으로
      연결될 수 있다는 가설을 검증한다.

두 파트로 구성 (기존 두 스크립트와 동일한 방법론 재사용, 타겟만 유동성으로 교체):

[파트 1: 정적 변수] gwr_feature_selection.py와 동일한 구조
  관측 단위: 단지 1개 = 1행 (316개 전체 — MDD처럼 최소거래건수 필터 없음. 유동성 자체를
             보려는 것이므로 거래가 적은 단지를 미리 걸러내면 안 됨)
  타겟:      trade_activity_ratio = 거래가 있었던 분기 수 / 전체 관측 분기 수
             (0~1, 높을수록 자주 거래됨=유동성 높음)
  검증:      일반 K-fold (단지당 1행이라 그룹 누수 없음)

[파트 2: 시변 변수] panel_feature_selection.py와 동일한 구조
  관측 단위: 단지×분기 전체 패널 (거래 없는 분기도 deal_count=0으로 포함 — 유동성 자체가
             "거래가 있었는지"이므로 가격이 있는 행만 쓰면 타겟 정의가 무너짐)
  타겟:      deal_count(결측=0) - 단지평균  (단지 고정효과 제거 후 거래량 편차)
  검증:      중첩 GroupKFold (단지 단위 완전분리)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINAL_CSV = "data/csv/final.csv"

SLOPE_RADII = [100, 150, 200, 300]
SLOPE_STATS = ["mean", "median", "stdev", "min", "max", "range"]
TRANSIT_COLS = ["straight_dist_m", "walk_dist_m", "walk_time_min", "avg_daily_ridership"]
STATION_ZONE_ORDER = {"초역세권": 1, "역세권": 2, "준역세권": 3, "비역세권": 4}

TIME_VARYING_COLS = ["commercial_index_lasso19", "population", "living_population", "pop_growth_mom"]

UNIT_COUNT_CSV = "data/csv/apt_unit_count_동대문구.csv"

N_SPLITS = 5
N_REPEATS = 20
LASSO_MAX_ITER = 300000
MIN_QUARTERS_PER_COMPLEX = 2


# ------------------------------------------------------------------
# 공용 유틸
# ------------------------------------------------------------------
def nested_kfold_r2(X: pd.DataFrame, y: pd.Series) -> float:
    scores = []
    for seed in range(N_REPEATS):
        outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, te_idx in outer.split(X):
            inner = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            model = make_pipeline(StandardScaler(), LassoCV(cv=inner, random_state=seed, max_iter=LASSO_MAX_ITER))
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            scores.append(model.score(X.iloc[te_idx], y.iloc[te_idx]))
    return float(np.mean(scores))


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
            model = make_pipeline(StandardScaler(), LassoCV(cv=inner_cv, max_iter=LASSO_MAX_ITER))
            model.fit(Xp.iloc[tr_idx], yp.iloc[tr_idx])
            scores.append(model.score(Xp.iloc[te_idx], yp.iloc[te_idx]))
    return float(np.mean(scores))


def fit_final_lasso(X: pd.DataFrame, y: pd.Series, cv) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LassoCV(cv=cv, max_iter=LASSO_MAX_ITER, alphas=200)
    model.fit(X_scaled, y)
    coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
    print(f"  alpha={model.alpha_:.4f}  R2(전체데이터 적합)={model.score(X_scaled, y):.3f}  n={len(y)}")
    selected = coefs[coefs != 0]
    print(f"  선택된 변수 {len(selected)}/{len(X.columns)}개:")
    for name, coef in selected.items():
        print(f"    {name:28s} {coef:+.4f}")
    return model, coefs


# ------------------------------------------------------------------
# 공용: 단지 1행 테이블에 대해 반경 선택 + 최종 LASSO 실행/저장
# ------------------------------------------------------------------
def run_radius_lasso(table: pd.DataFrame, other_features: list, target_col: str, tag: str):
    radius_results = []
    for radius in SLOPE_RADII:
        slope_cols = [f"slope_{stat}_{radius}m" for stat in SLOPE_STATS]
        feature_cols = slope_cols + other_features
        sub = table.dropna(subset=feature_cols + [target_col])
        r2 = nested_kfold_r2(sub[feature_cols], sub[target_col])
        radius_results.append({"radius_m": radius, "nested_cv_r2": r2, "n": len(sub)})
        print(f"  반경 {radius}m: 중첩 K-fold R2={r2:.4f} (n={len(sub)})")

    best_radius = max(radius_results, key=lambda d: d["nested_cv_r2"])["radius_m"]
    print(f"  -> 채택 반경: {best_radius}m")

    slope_cols = [f"slope_{stat}_{best_radius}m" for stat in SLOPE_STATS]
    feature_cols = slope_cols + other_features
    sub = table.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"\n[최종 LASSO] 반경 {best_radius}m + 나머지 {len(other_features)}개 변수 -> {target_col}")
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    _, coefs = fit_final_lasso(sub[feature_cols], sub[target_col], cv)

    pd.DataFrame(radius_results).to_csv(f"data/csv/liquidity_{tag}_radius_comparison.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv(f"data/csv/liquidity_{tag}_lasso_coefs.csv", encoding="utf-8-sig")
    table.to_csv(f"data/csv/liquidity_{tag}_table.csv", index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------
# 세대수(단지 규모) 로드 — 한국부동산원 공동주택 단지 식별정보(세대수_크롤링.py 산출물)
# ------------------------------------------------------------------
# 이 API의 COMPLEX_PK 체계가 final.csv의 complex_id와 달라 직접 조인 불가 -> 주소(동+지번)로 매칭.
# 336개 아파트 중 305개(95.6%) 매칭 성공, 나머지 14개는 최근 재개발 대단지 등 API 데이터에
# 아직 없는 케이스 (예: e편한세상답십리아르테포레, 청량리역롯데캐슬SKY-L65) — dropna로 자동 제외됨.
def load_unit_counts() -> pd.DataFrame:
    units = pd.read_csv(UNIT_COUNT_CSV)
    units = units.dropna(subset=["dong", "jibun"])
    units["jibun"] = units["jibun"].astype(str)
    units = units.drop_duplicates(subset=["dong", "jibun"])
    return units[["dong", "jibun", "UNIT_CNT"]].rename(columns={"UNIT_CNT": "unit_cnt"})


# ------------------------------------------------------------------
# 단지 1행 테이블 로드 (활동비율 타겟 + 정적 후보 변수 + 세대수)
# ------------------------------------------------------------------
def build_static_table() -> pd.DataFrame:
    df = pd.read_csv(FINAL_CSV)
    df["traded"] = df["deal_count"].fillna(0) > 0

    activity = df.groupby("complex_id").agg(
        trade_activity_ratio=("traded", "mean"),
        n_quarters=("traded", "size"),
    ).reset_index()

    static_cols = [f"slope_{stat}_{r}m" for r in SLOPE_RADII for stat in SLOPE_STATS]
    static_cols += ["school_pc1"] + TRANSIT_COLS + ["station_zone", "dong", "jibun"]
    static = df.groupby("complex_id")[static_cols].first().reset_index()
    static["station_zone_ord"] = static["station_zone"].map(STATION_ZONE_ORDER)
    static = static.drop(columns=["station_zone"])
    static["jibun"] = static["jibun"].astype(str)

    unit_counts = load_unit_counts()
    static = static.merge(unit_counts, on=["dong", "jibun"], how="left")
    static = static.drop(columns=["dong", "jibun"])

    table = static.merge(activity, on="complex_id", how="inner")
    return table


# ------------------------------------------------------------------
# 파트 1: 정적 변수 -> trade_activity_ratio
# ------------------------------------------------------------------
def run_static_part():
    print(f"\n{'='*70}\n[파트 1: 정적 입지변수만] 타겟 = trade_activity_ratio (거래분기비율)\n{'='*70}")

    table = build_static_table()
    print(f"[병합] {len(table)}개 단지, trade_activity_ratio 분포: "
          f"mean={table['trade_activity_ratio'].mean():.3f}, std={table['trade_activity_ratio'].std():.3f}")

    other_features = ["school_pc1"] + TRANSIT_COLS + ["station_zone_ord", "unit_cnt"]
    run_radius_lasso(table, other_features, "trade_activity_ratio", tag="static")


# ------------------------------------------------------------------
# 파트 1-보강: 정적 변수 + 시변 변수의 "단지별 시간평균" -> trade_activity_ratio
# ------------------------------------------------------------------
# 시변 변수(상권/인구)는 원래 분기마다 값이 바뀌지만, 여기서는 타겟(trade_activity_ratio)
# 자체가 단지 1개=1행(패널이 아님)이라 데멘(단지 고정효과 제거) 문제가 발생하지 않는다.
# 그래서 "이 단지가 7년간 평균적으로 어떤 상권/인구 수준이었는가"를 단지별 시간평균으로
# 접어서 정적 변수와 같은 테이블에 나란히 투입할 수 있다 — 지금까지 안 해본 조합.
def run_combined_part():
    print(f"\n{'='*70}\n[파트 1-보강: 정적+시변 시간평균] 타겟 = trade_activity_ratio\n{'='*70}")

    static_table = build_static_table()

    df = pd.read_csv(FINAL_CSV)
    time_avg = df.groupby("complex_id")[TIME_VARYING_COLS].mean().reset_index()
    time_avg.columns = ["complex_id"] + [f"{c}_avg" for c in TIME_VARYING_COLS]

    table = static_table.merge(time_avg, on="complex_id", how="inner")
    time_avg_cols = [f"{c}_avg" for c in TIME_VARYING_COLS]
    print(f"[병합] {len(table)}개 단지, 정적 변수 + 시변 시간평균 {len(time_avg_cols)}개 추가")

    other_features = ["school_pc1"] + TRANSIT_COLS + ["station_zone_ord", "unit_cnt"] + time_avg_cols
    run_radius_lasso(table, other_features, "trade_activity_ratio", tag="combined")


# ------------------------------------------------------------------
# 파트 2: 시변 변수 -> deal_count 편차
# ------------------------------------------------------------------
def run_timevarying_part():
    print(f"\n{'='*70}\n[파트 2: 시변 입지변수] 타겟 = deal_count 편차 (단지 고정효과 제거)\n{'='*70}")

    df = pd.read_csv(FINAL_CSV)
    df["deal_count"] = df["deal_count"].fillna(0)

    n_quarters = df.groupby("complex_id")["yyyymm"].transform("size")
    df = df[n_quarters >= MIN_QUARTERS_PER_COMPLEX].copy()

    df["deal_count_deviation"] = df["deal_count"] - df.groupby("complex_id")["deal_count"].transform("mean")

    print(f"[패널] {len(df)}행, {df['complex_id'].nunique()}개 단지 "
          f"(거래 없는 분기도 deal_count=0으로 포함)")

    sub = df.dropna(subset=TIME_VARYING_COLS + ["deal_count_deviation"]).reset_index(drop=True)
    X, y, groups = sub[TIME_VARYING_COLS], sub["deal_count_deviation"], sub["complex_id"]

    r2 = nested_groupkfold_r2(X, y, groups)
    print(f"  중첩 GroupKFold R2={r2:.4f} (n={len(sub)}, 단지수={groups.nunique()})")

    print(f"\n[최종 LASSO] 시변 변수 {len(TIME_VARYING_COLS)}개 -> deal_count_deviation")
    cv = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))
    _, coefs = fit_final_lasso(X, y, cv)

    pd.DataFrame([{"target": "deal_count_deviation", "nested_groupkfold_r2": r2, "n": len(sub)}]).to_csv(
        "data/csv/liquidity_panel_result_summary.csv", index=False, encoding="utf-8-sig")
    coefs.to_csv("data/csv/liquidity_panel_lasso_coefs.csv", encoding="utf-8-sig")


def main():
    run_static_part()
    run_combined_part()
    run_timevarying_part()
    print("\n[저장] data/csv/liquidity_static_*.csv, data/csv/liquidity_combined_*.csv, data/csv/liquidity_panel_*.csv")


if __name__ == "__main__":
    main()
