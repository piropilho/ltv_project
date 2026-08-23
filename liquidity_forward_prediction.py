"""
미래 유동성 예측 — "매수 후 1년 뒤 유동성이 어떻게 될까"
================================================================
지금까지 만든 유동성 모델(liquidity_feature_selection.py, liquidity_xgboost_check.py)은
"단지 1개=1행, 7년 전체 누적 요약"이라 오늘 시점 리스크 설명용일 뿐 미래예측이 아니다.
이 스크립트는 실수요자 시나리오("답십리대우아파트를 오늘 사면 1년 뒤 유동성은?")에 맞춰
진짜 미래예측 모델을 만든다.

관측 단위: 단지×분기(t) 패널 (t = "오늘 산다고 가정하는 시점")
타겟:      forward_activity_ratio_1y(t) = t+1~t+4분기(향후 1년) 중 거래 있었던 분기 비율
피처:      t 시점까지만 알 수 있는 정보만 사용 (미래 누수 방지) —
           - unit_cnt, slope_mean_300m 등 경사도 통계량 (정적)
           - population_t, living_population_t (7년 평균이 아니라 t 시점 당해 값)
           - trailing_activity_ratio(t, N) = t 이전 N분기 동안의 거래활동비율 (모멘텀 피처)
검증:      walk-forward (분기 t 기준 시간분할, 트랙B와 동일 원칙 — 랜덤분할/K-fold 금지)

트레일링 윈도우(N): 4/8/12분기 비교 결과 4분기 채택 (12분기와 사실상 동률이었으나
표본이 훨씬 많고 예측기간(1년)과 대칭이라 실무적으로 우위) -- weighted wf R2=0.636

이후 R2를 0.7까지 올리기 위한 피처 추가 실험 로그 (STEP 순서대로 검증, 각 스텝은
직전 채택된 피처셋 위에 하나씩 추가 -- 결과는 data/csv/liquidity_forward_step_comparison.csv):
  STEP 0  베이스라인                                      R2=0.636
  STEP 1  +commercial_index_lasso19_t, +pop_growth_mom_t  R2=0.641  [채택] 미미하지만 해 없음
  STEP 2  +시장전체 국면 피처(market_trailing_activity_t)  R2=0.592  [기각] walk-forward 테스트
          구간이 학습 때 못 본 시장국면(2022년 거래절벽)일 때 트리모델이 외삽 못 해 악화
  STEP 3  +12분기 트레일링 병행 투입                        R2=0.589  [기각] dropna로 초반 t가
          날아가 학습표본이 5배 줄어드는 손실이 신호 이득보다 큼
  STEP 4  +price_momentum (avg_price_per_m2 재가공, 신규 데이터 없음)  R2=0.654  [채택]
  STEP 5  +quarter_of_year(계절성 더미)                    R2=0.650  [기각] 사실상 무효과
  STEP 6  +days_since_last_tightening_t, +n_tightening_past_1y_t     R2=0.687  [채택]
          트랙B의 policy_shock_features(build_master_data.POLICY_EVENTS) 재사용. STEP2와 달리
          "이벤트로부터 경과일"이라는 bounded 피처라 walk-forward 외삽 문제 없이 전 폴드
          고르게 개선됨 (특히 후반 폴드 0.674->0.750로 크게 개선)
  STEP 7  +jeonse_ratio (전세가율, 전월세_크롤링.py 신규 크롤링 — 2순위)      R2=0.680  [기각]
          소폭 악화(-0.007). 순수전세 거래가 없는 분기가 있는 단지들이 dropna로 빠지며
          표본이 줄어든 손실이 신호 이득을 상쇄한 것으로 보임
  -> 현재 채택된 최종 피처셋 = STEP1 + STEP4 + STEP6
     (BASELINE_FEATURES + t시점 피처 2개 + price_momentum + 정책이벤트 피처 2개)
     최종 weighted walk-forward R2 = 0.687 (목표 0.7 근접, 2순위 중 전세가율은 기각 -- 남은
     후보는 재건축/정비사업 진행단계, 비용 대비 효과 판단 후 착수 여부 결정 예정)
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from liquidity_feature_selection import FINAL_CSV, load_unit_counts, STATION_ZONE_ORDER
from policy_events import policy_shock_features

TRAILING_CANDIDATES = [4, 8, 12]  # 분기 단위 (1년/2년/3년)
FORWARD_QUARTERS = 4              # 예측 대상: 향후 1년(4분기)
BEST_RADIUS = 300                 # liquidity_feature_selection.py에서 채택된 반경 재사용
STATIC_SLOPE_COLS = [f"slope_{stat}_{BEST_RADIUS}m" for stat in
                      ["mean", "median", "stdev", "min", "max", "range"]]
N_FOLDS = 4

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

JEONSE_CSV = "data/csv/전월세_2019_2025_동대문구.csv"
MONTH_TO_QLABEL = {1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6, 7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12}


def load_jeonse_price(q_index: dict) -> pd.DataFrame:
    """(dong, jibun, t_idx) -> 순수전세 평균 m2당 보증금. 전월세_크롤링.py 산출물 사용."""
    rent = pd.read_csv(JEONSE_CSV)
    rent["monthlyRent"] = rent["monthlyRent"].astype(str).str.replace(",", "").astype(float)
    rent = rent[rent["monthlyRent"] == 0].copy()  # 순수전세만 (월세 섞이면 보증금 단순비교 불가)

    rent["deposit"] = rent["deposit"].astype(str).str.replace(",", "").astype(float)
    rent["excluUseAr"] = rent["excluUseAr"].astype(float)
    rent["jeonse_price_per_m2"] = rent["deposit"] / rent["excluUseAr"]

    # final.csv와 동일한 분기 라벨링(1~3월->03, 4~6->06, 7~9->09, 10~12->12)으로 맞춰야 t_idx 매칭 가능
    rent["yyyymm"] = rent["dealYear"].astype(int) * 100 + rent["dealMonth"].astype(int).map(MONTH_TO_QLABEL)
    rent = rent[rent["yyyymm"].isin(q_index)].copy()
    rent["t_idx"] = rent["yyyymm"].map(q_index)
    rent["jibun"] = rent["jibun"].astype(str)

    agg = rent.groupby(["umdNm", "jibun", "t_idx"], as_index=False)["jeonse_price_per_m2"].mean()
    return agg.rename(columns={"umdNm": "dong"})


def load_base_panel() -> tuple[pd.DataFrame, list]:
    df = pd.read_csv(FINAL_CSV)
    df["deal_count"] = df["deal_count"].fillna(0)

    quarters = sorted(df["yyyymm"].unique())
    q_index = {q: i for i, q in enumerate(quarters)}
    df["t_idx"] = df["yyyymm"].map(q_index)

    # complex_id가 물리적으로 여러 동(예: 휘경주공1/2단지)에 걸쳐 중복되는 케이스가 있어
    # (complex_id, t_idx) 기준으로 한 번 더 접어 유일하게 만든다 -- 아니면 뒤에서
    # g.loc[t, col] 스칼라 조회가 깨짐.
    quarterly = df.groupby(["complex_id", "t_idx"], as_index=False).agg(
        deal_count=("deal_count", "sum"),
        population=("population", "first"),
        living_population=("living_population", "first"),
        commercial_index_lasso19=("commercial_index_lasso19", "first"),
        pop_growth_mom=("pop_growth_mom", "first"),
        avg_price_per_m2=("avg_price_per_m2", "mean"),
        dong=("dong", "first"),
        jibun=("jibun", "first"),
    )
    quarterly["traded"] = quarterly["deal_count"] > 0
    quarterly["jibun"] = quarterly["jibun"].astype(str)

    jeonse = load_jeonse_price(q_index)
    quarterly = quarterly.merge(jeonse, on=["dong", "jibun", "t_idx"], how="left")
    quarterly = quarterly.drop(columns=["dong", "jibun"])

    # 거래 없는 분기는 가격이 NaN이므로, 마지막 관측 거래가를 그대로 이어붙인다
    # (그 단지의 "현재 알려진 시세"를 나타냄 -- 표준적인 비유동자산 가격 처리 방식)
    quarterly = quarterly.sort_values(["complex_id", "t_idx"])
    quarterly["price_ffill"] = quarterly.groupby("complex_id")["avg_price_per_m2"].ffill()
    quarterly["jeonse_price_ffill"] = quarterly.groupby("complex_id")["jeonse_price_per_m2"].ffill()

    static_cols = STATIC_SLOPE_COLS + ["dong", "jibun"]
    static = df.groupby("complex_id")[static_cols].first().reset_index()
    static["jibun"] = static["jibun"].astype(str)
    units = load_unit_counts()
    static = static.merge(units, on=["dong", "jibun"], how="left").drop(columns=["dong", "jibun"])

    panel = quarterly.merge(static, on="complex_id", how="left")
    return panel, quarters


QUARTER_OF_YEAR = {3: 1, 6: 2, 9: 3, 12: 4}  # yyyymm의 월(3/6/9/12) -> 분기순번(이사철 계절성)


def build_panel_for_window(df: pd.DataFrame, n_quarters: int, trailing_n: int,
                            quarters: list | None = None) -> pd.DataFrame:
    """단지별로 (t, trailing_activity_ratio(t), forward_activity_ratio_1y(t)) 생성."""
    rows = []
    for complex_id, g in df.groupby("complex_id"):
        g = g.sort_values("t_idx").set_index("t_idx")
        traded = g["traded"]
        price = g["price_ffill"]
        jeonse_price = g["jeonse_price_ffill"]
        for t in range(trailing_n - 1, n_quarters - FORWARD_QUARTERS):
            trailing_window = traded.reindex(range(t - trailing_n + 1, t + 1))
            forward_window = traded.reindex(range(t + 1, t + 1 + FORWARD_QUARTERS))
            if trailing_window.isna().any() or forward_window.isna().any():
                continue

            price_t = price.get(t, np.nan)
            price_lag = price.get(t - trailing_n, np.nan)
            price_momentum = (price_t / price_lag - 1) if pd.notna(price_t) and pd.notna(price_lag) and price_lag != 0 else np.nan

            jeonse_t = jeonse_price.get(t, np.nan)
            jeonse_ratio = (jeonse_t / price_t) if pd.notna(jeonse_t) and pd.notna(price_t) and price_t != 0 else np.nan

            row = {
                "complex_id": complex_id,
                "t_idx": t,
                "trailing_activity_ratio": trailing_window.mean(),
                "forward_activity_ratio_1y": forward_window.mean(),
                "population_t": g.loc[t, "population"],
                "living_population_t": g.loc[t, "living_population"],
                "commercial_index_lasso19_t": g.loc[t, "commercial_index_lasso19"],
                "pop_growth_mom_t": g.loc[t, "pop_growth_mom"],
                "unit_cnt": g.loc[t, "unit_cnt"],
                "price_momentum": price_momentum,
                "jeonse_ratio": jeonse_ratio,
            }
            if quarters is not None:
                row["quarter_of_year"] = QUARTER_OF_YEAR[quarters[t] % 100]
                as_of_date = pd.to_datetime(str(quarters[t]), format="%Y%m")
                days_since, n_recent = policy_shock_features(as_of_date)
                row["days_since_last_tightening_t"] = days_since
                row["n_tightening_past_1y_t"] = n_recent
            for c in STATIC_SLOPE_COLS:
                row[c] = g.loc[t, c]
            rows.append(row)
    return pd.DataFrame(rows)


def walkforward_r2(panel: pd.DataFrame, feature_cols: list, target_col: str) -> tuple[float, pd.DataFrame]:
    t_values = sorted(panel["t_idx"].unique())
    warmup = max(1, int(len(t_values) * 0.5))
    test_t_values = t_values[warmup:]
    if not test_t_values:
        return float("nan"), pd.DataFrame()

    fold_edges = np.array_split(test_t_values, min(N_FOLDS, len(test_t_values)))
    fold_results = []
    for fold_i, test_ts in enumerate(fold_edges):
        test_ts = list(test_ts)
        train_ts = [t for t in t_values if t < min(test_ts)]
        train = panel[panel["t_idx"].isin(train_ts)].dropna(subset=feature_cols + [target_col])
        test = panel[panel["t_idx"].isin(test_ts)].dropna(subset=feature_cols + [target_col])
        if len(train) < 20 or len(test) < 5:
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[feature_cols], train[target_col])
        pred = model.predict(test[feature_cols])
        ss_res = ((test[target_col] - pred) ** 2).sum()
        ss_tot = ((test[target_col] - test[target_col].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        fold_results.append({"fold": fold_i, "test_t_range": f"{min(test_ts)}~{max(test_ts)}",
                              "n_train": len(train), "n_test": len(test), "r2": r2})

    fold_df = pd.DataFrame(fold_results)
    if fold_df.empty:
        return float("nan"), fold_df
    weighted_r2 = float(np.average(fold_df["r2"], weights=fold_df["n_test"]))
    return weighted_r2, fold_df


BASELINE_TRAILING_N = 4  # 앞 단계에서 채택된 참조기간
BASELINE_FEATURES = ["trailing_activity_ratio", "unit_cnt", "population_t", "living_population_t"] + STATIC_SLOPE_COLS


def main():
    base_panel, quarters = load_base_panel()
    n_quarters = len(quarters)
    print(f"[패널] 전체 {n_quarters}개 분기 ({quarters[0]}~{quarters[-1]}), "
          f"{base_panel['complex_id'].nunique()}개 단지")

    panel = build_panel_for_window(base_panel, n_quarters, BASELINE_TRAILING_N, quarters)
    target_col = "forward_activity_ratio_1y"

    print(f"\n{'='*70}\n[STEP 0: 베이스라인] 참조기간 {BASELINE_TRAILING_N}분기, 피처 {len(BASELINE_FEATURES)}개\n{'='*70}")
    r2_base, fold_df = walkforward_r2(panel, BASELINE_FEATURES, target_col)
    print(fold_df.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_base:.4f}")

    print(f"\n{'='*70}\n[STEP 1: 누락된 t시점 피처 추가] +commercial_index_lasso19_t, +pop_growth_mom_t\n{'='*70}")
    step1_features = BASELINE_FEATURES + ["commercial_index_lasso19_t", "pop_growth_mom_t"]
    r2_step1, fold_df1 = walkforward_r2(panel, step1_features, target_col)
    print(fold_df1.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step1:.4f}  (베이스라인 대비 {r2_step1 - r2_base:+.4f})")

    # STEP 2(시장 전체 국면 피처)는 기각됨: walk-forward 테스트 구간이 학습 때 못 본
    # 시장국면(예: 2022년 거래절벽)일 때 트리모델이 외삽을 못 해 오히려 악화(-0.048).
    # 자세한 진단은 대화 기록 참고. STEP1까지만 채택하고 STEP3로 진행.

    print(f"\n{'='*70}\n[STEP 3: 트레일링 윈도우 이중 투입] +trailing_activity_ratio_12q\n{'='*70}")
    panel_12q = build_panel_for_window(base_panel, n_quarters, 12)[["complex_id", "t_idx", "trailing_activity_ratio"]]
    panel_12q = panel_12q.rename(columns={"trailing_activity_ratio": "trailing_activity_ratio_12q"})
    panel_step3 = panel.merge(panel_12q, on=["complex_id", "t_idx"], how="left")
    step3_features = step1_features + ["trailing_activity_ratio_12q"]
    r2_step3, fold_df3 = walkforward_r2(panel_step3, step3_features, target_col)
    print(fold_df3.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step3:.4f}  (STEP1 대비 {r2_step3 - r2_step1:+.4f})")

    # STEP 3(12분기 트레일링 이중투입)도 기각됨: dropna로 학습표본이 5배 줄어드는 부작용이
    # 신호 이득보다 커서 악화(-0.052). STEP1 피처셋(panel) 기준으로 계속 진행.

    print(f"\n{'='*70}\n[STEP 4: 가격 모멘텀 추가] +price_momentum (0순위, 신규 데이터 없이 기존 avg_price_per_m2 재가공)\n{'='*70}")
    step4_features = step1_features + ["price_momentum"]
    r2_step4, fold_df4 = walkforward_r2(panel, step4_features, target_col)
    print(fold_df4.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step4:.4f}  (STEP1 대비 {r2_step4 - r2_step1:+.4f})")

    print(f"\n{'='*70}\n[STEP 5: 계절성 더미 추가] +quarter_of_year\n{'='*70}")
    step5_features = step4_features + ["quarter_of_year"]
    r2_step5, fold_df5 = walkforward_r2(panel, step5_features, target_col)
    print(fold_df5.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step5:.4f}  (STEP4 대비 {r2_step5 - r2_step4:+.4f})")

    # STEP 5(계절성 더미)도 기각됨: 사실상 무효과(-0.004). STEP4 피처셋(price_momentum
    # 포함) 기준으로 계속 진행.

    print(f"\n{'='*70}\n[STEP 6: 정책규제 이벤트 캘린더 추가] +days_since_last_tightening_t, +n_tightening_past_1y_t "
          f"(1순위, 트랙B의 policy_shock_features 재사용)\n{'='*70}")
    step6_features = step4_features + ["days_since_last_tightening_t", "n_tightening_past_1y_t"]
    r2_step6, fold_df6 = walkforward_r2(panel, step6_features, target_col)
    print(fold_df6.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step6:.4f}  (STEP4 대비 {r2_step6 - r2_step4:+.4f})")

    print(f"\n{'='*70}\n[STEP 7: 전세가율 추가] +jeonse_ratio (2순위, 전월세_크롤링.py 신규 데이터)\n{'='*70}")
    step7_features = step6_features + ["jeonse_ratio"]
    r2_step7, fold_df7 = walkforward_r2(panel, step7_features, target_col)
    print(fold_df7.to_string(index=False))
    print(f"  -> 가중평균 walk-forward R2 = {r2_step7:.4f}  (STEP6 대비 {r2_step7 - r2_step6:+.4f})")

    pd.DataFrame([
        {"step": "0_baseline", "n_features": len(BASELINE_FEATURES), "weighted_wf_r2": r2_base},
        {"step": "1_add_t_features", "n_features": len(step1_features), "weighted_wf_r2": r2_step1},
        {"step": "2_add_market_feature(기각)", "n_features": len(step1_features) + 1, "weighted_wf_r2": None},
        {"step": "3_add_12q_trailing(기각)", "n_features": len(step3_features), "weighted_wf_r2": r2_step3},
        {"step": "4_add_price_momentum", "n_features": len(step4_features), "weighted_wf_r2": r2_step4},
        {"step": "5_add_seasonality(기각)", "n_features": len(step5_features), "weighted_wf_r2": r2_step5},
        {"step": "6_add_policy_events", "n_features": len(step6_features), "weighted_wf_r2": r2_step6},
        {"step": "7_add_jeonse_ratio", "n_features": len(step7_features), "weighted_wf_r2": r2_step7},
    ]).to_csv("data/csv/liquidity_forward_step_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n[저장] data/csv/liquidity_forward_step_comparison.csv")


if __name__ == "__main__":
    main()
