"""
단지별 LTV 차등 부여 — "정부 상한 이내에서, 안전한 담보는 상한까지 완화해주자"
=================================================================================
프로젝트 목적: LTV 규제 자체를 바꾸거나 우회하는 게 아니라, 정부가 정한 LTV 상한
"이내에서" 은행이 실제로 얼마까지 빌려줄지를 정교화하는 모델. 은행이 이미 개인
신용점수로 상한 내 차등 승인하듯(예: 상한 40%인데 신용점수 낮으면 30%만), 여기에
"담보물의 공간적 리스크(=유동성)"라는 축을 하나 더 추가한다. 안전한(유동성 높은)
단지는 상한에 최대한 가깝게 완화해주고, 위험한(유동성 낮은) 단지는 보수적으로 낮춘다.

법적 근거: 2025.10.15 대책(부동산관계장관회의)으로 서울 전역(동대문구 포함)이
투기과열지구+조정대상지역으로 동시 지정되어, 무주택 실수요자 기준 LTV 상한이
40%로 확정되었다 (POLICY_EVENTS의 "10.15대책(2025)"과 동일 사건). 이 40%를
LTV_CAP으로 채택 — 프로젝트 기획안이 예시로 든 수치와 정확히 일치.

리스크 지표: liquidity_forward_prediction.py가 검증한 "1년 후 유동성" 예측 모델
(walk-forward R2=0.687, STEP1+STEP4+STEP6 피처셋)을 전체 데이터로 재학습한 뒤,
가장 최근 분기(2025-12) 시점에서 각 단지의 향후 1년 유동성을 예측한다. 이 예측값의
백분위를 4분위로 나눠 LTV를 차등 부여한다.

주의(외삽 방지): 정책이벤트 경과일(days_since_last_tightening) 계산 시 실제 "오늘"
날짜가 아니라 데이터의 마지막 관측 분기(2025-12) 시점을 기준으로 삼는다. 학습 데이터의
날짜 범위를 벗어나면 트리모델이 외삽하지 못해 성능이 저하된다는 게 이미
liquidity_forward_prediction.py STEP2에서 확인된 문제라, 동일한 리스크를 피하기 위함.

차등 구조: 4분위 이산 티어 대신 백분위 연속 선형 스케일링을 채택. 원래 4분위
이산 등급(1~4등급)으로 설계했었는데, "이게 실제 여신심사 관행과 일치하냐"는
질문에 확인해보니 안 맞았다 -- 한국 개인신용평가는 2021년 신용점수제 전환 이후
1~10등급 같은 이산 등급이 아니라 0~1000점 연속 점수로 대출조건을 차등한다.
그래서 "신용점수제와 같은 원리"라는 설명이 실제로 성립하도록, 예측 유동성의
백분위(0~100)를 LTV_FLOOR~LTV_CAP 구간에 그대로 선형 매핑한다. 이산 등급 대비
개별 단지 간 미세한 차이도 반영되지만, "몇 등급"이라는 직관적 설명력은 줄어든다.
정확한 상한/폭 수치는 팀 협의로 조정 가능하도록 상수로 노출.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        matplotlib.rc("font", family=font_name)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

from liquidity_feature_selection import FINAL_CSV
from liquidity_forward_prediction import (
    load_base_panel, build_panel_for_window, BASELINE_TRAILING_N, BASELINE_FEATURES,
    XGB_PARAMS, STATIC_SLOPE_COLS,
)
from policy_events import policy_shock_features

TARGET_COL = "forward_activity_ratio_1y"
STEP6_EXTRA_FEATURES = ["commercial_index_lasso19_t", "pop_growth_mom_t", "price_momentum",
                         "days_since_last_tightening_t", "n_tightening_past_1y_t"]
FEATURE_COLS = BASELINE_FEATURES + STEP6_EXTRA_FEATURES

# ------------------------------------------------------------------
# LTV 매핑 상수 — 근거는 모듈 docstring 참고, 팀 협의로 조정 가능
# ------------------------------------------------------------------
LTV_CAP = 40.0             # 무주택 실수요자 기준 정부 상한 (2025.10.15 대책, 동대문구 투기과열+조정대상)
LTV_MAX_DISCOUNT = 10.0    # 최대 할인폭(%p) -- 기획안의 신용점수 차등 예시(40%->30%)와 동일 폭
LTV_FLOOR = LTV_CAP - LTV_MAX_DISCOUNT   # 예측 유동성이 가장 낮은 단지의 LTV (=30.0)
OUT_PNG = "data/img/ltv_assignment.png"


def train_final_model(base_panel: pd.DataFrame, quarters: list) -> xgb.XGBRegressor:
    """검증 완료된 STEP1+4+6 피처셋(walk-forward R2=0.687)으로 전체 이력 데이터를 재학습."""
    n_quarters = len(quarters)
    panel = build_panel_for_window(base_panel, n_quarters, BASELINE_TRAILING_N, quarters)
    sub = panel.dropna(subset=FEATURE_COLS + [TARGET_COL])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(sub[FEATURE_COLS], sub[TARGET_COL])
    print(f"[최종모델 재학습] n={len(sub)} (전체 이력), 피처 {len(FEATURE_COLS)}개")
    return model


def build_latest_snapshot(base_panel: pd.DataFrame, quarters: list) -> pd.DataFrame:
    """가장 최근 분기(t=마지막) 기준 단지별 예측용 피처 스냅샷.
    forward_activity_ratio_1y(향후 1년)는 아직 관측되지 않은 미래이므로 라벨 없이,
    build_panel_for_window와 동일한 피처 산출 로직만 t=마지막 분기 1개에 대해 재사용한다."""
    n_quarters = len(quarters)
    t = n_quarters - 1
    as_of_date = pd.to_datetime(str(quarters[t]), format="%Y%m")
    trailing_n = BASELINE_TRAILING_N

    rows = []
    for complex_id, g in base_panel.groupby("complex_id"):
        g = g.sort_values("t_idx").set_index("t_idx")
        traded = g["traded"]
        price = g["price_ffill"]

        trailing_window = traded.reindex(range(t - trailing_n + 1, t + 1))
        if trailing_window.isna().any() or t not in g.index:
            continue

        price_t = price.get(t, np.nan)
        price_lag = price.get(t - trailing_n, np.nan)
        price_momentum = (price_t / price_lag - 1) if pd.notna(price_t) and pd.notna(price_lag) and price_lag != 0 else np.nan

        days_since, n_recent = policy_shock_features(as_of_date)

        row = {
            "complex_id": complex_id,
            "trailing_activity_ratio": trailing_window.mean(),
            "population_t": g.loc[t, "population"],
            "living_population_t": g.loc[t, "living_population"],
            "commercial_index_lasso19_t": g.loc[t, "commercial_index_lasso19"],
            "pop_growth_mom_t": g.loc[t, "pop_growth_mom"],
            "unit_cnt": g.loc[t, "unit_cnt"],
            "price_momentum": price_momentum,
            "days_since_last_tightening_t": days_since,
            "n_tightening_past_1y_t": n_recent,
        }
        for c in STATIC_SLOPE_COLS:
            row[c] = g.loc[t, c]
        rows.append(row)

    snapshot = pd.DataFrame(rows)
    print(f"[스냅샷] 기준시점={quarters[t]}, {len(snapshot)}개 단지 (트레일링 {trailing_n}분기 데이터 확보된 단지만)")
    return snapshot


def assign_ltv(snapshot: pd.DataFrame, model: xgb.XGBRegressor) -> pd.DataFrame:
    sub = snapshot.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    dropped = len(snapshot) - len(sub)
    if dropped:
        print(f"[제외] 피처 결측으로 {dropped}개 단지 제외 (세대수 미매칭 등)")

    # 회귀 예측이라 [0,1] 경계(활동비율의 정의역)를 살짝 벗어날 수 있어 클리핑
    # (백분위 순위에는 영향 없음 — 단조 변환이라 상대순서 보존)
    sub["predicted_liquidity_1y"] = model.predict(sub[FEATURE_COLS]).clip(0.0, 1.0)
    sub["liquidity_percentile"] = sub["predicted_liquidity_1y"].rank(pct=True) * 100

    # 백분위(0~100)를 LTV_FLOOR~LTV_CAP 구간에 그대로 선형 매핑 (신용점수제와 동일 원리)
    sub["assigned_ltv_pct"] = (LTV_FLOOR + (LTV_CAP - LTV_FLOOR) * sub["liquidity_percentile"] / 100).round(1)
    return sub


def plot_result(result: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.get_cmap("RdYlGn")  # 빨강(위험/저유동성) -> 초록(안전/고유동성), 연속 색상

    ax = axes[0]
    sorted_r = result.sort_values("predicted_liquidity_1y").reset_index(drop=True)
    colors = cmap(sorted_r["liquidity_percentile"] / 100)
    ax.bar(range(len(sorted_r)), sorted_r["predicted_liquidity_1y"], color=colors, width=1.0)
    ax.set_xlabel("단지 (예측 유동성 오름차순 정렬)")
    ax.set_ylabel("예측 유동성 (향후 1년 거래분기비율)")
    ax.set_title("단지별 예측 유동성 분포\n(색: 백분위 낮음=빨강 -> 높음=초록)")

    ax = axes[1]
    x_line = np.linspace(0, 100, 100)
    y_line = LTV_FLOOR + (LTV_CAP - LTV_FLOOR) * x_line / 100
    ax.plot(x_line, y_line, color="#333333", linewidth=1.5, zorder=1, label="LTV 매핑 함수")
    ax.scatter(result["liquidity_percentile"], result["assigned_ltv_pct"],
               c=result["liquidity_percentile"], cmap=cmap, s=14, zorder=2, edgecolors="none")
    ax.axhline(LTV_CAP, color="#333333", linestyle="--", linewidth=1)
    ax.axhline(LTV_FLOOR, color="#333333", linestyle="--", linewidth=1)
    ax.text(2, LTV_CAP + 0.6, f"정부 상한 {LTV_CAP:.0f}%", fontsize=8, ha="left", color="#333333")
    ax.text(2, LTV_FLOOR - 1.6, f"최저 LTV {LTV_FLOOR:.0f}%", fontsize=8, ha="left", color="#333333")
    ax.set_xlabel("예측 유동성 백분위 (동대문구 내 상대순위)")
    ax.set_ylabel("승인 LTV (%)")
    ax.set_ylim(LTV_FLOOR - 4, LTV_CAP + 4)
    ax.set_title("백분위 -> LTV 연속 선형 매핑\n(신용점수제와 동일 원리: 등급이 아니라 연속 점수로 차등)")

    fig.suptitle("동대문구 단지별 예측 유동성 기반 LTV 차등 부여 (연속 백분위 방식)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"[저장] {OUT_PNG}")


def main():
    base_panel, quarters = load_base_panel()

    model = train_final_model(base_panel, quarters)
    snapshot = build_latest_snapshot(base_panel, quarters)
    result = assign_ltv(snapshot, model)

    names = pd.read_csv(FINAL_CSV)[["complex_id", "apt_name"]].drop_duplicates("complex_id")
    result = result.merge(names, on="complex_id", how="left")

    print(f"\n[백분위 구간별 요약] (참고용 -- 실제 LTV는 연속값이라 구간 구분 없이 산출됨)")
    result["_quartile_view"] = pd.qcut(result["liquidity_percentile"], 4,
                                        labels=["하위 25%", "중하 25%", "중상 25%", "상위 25%"])
    summary = result.groupby("_quartile_view", observed=True).agg(
        n_complex=("complex_id", "size"),
        mean_predicted_liquidity=("predicted_liquidity_1y", "mean"),
        mean_assigned_ltv_pct=("assigned_ltv_pct", "mean"),
        min_ltv=("assigned_ltv_pct", "min"),
        max_ltv=("assigned_ltv_pct", "max"),
    ).reset_index()
    print(summary.to_string(index=False))
    result = result.drop(columns=["_quartile_view"])

    out_cols = ["complex_id", "apt_name", "unit_cnt", "predicted_liquidity_1y", "liquidity_percentile",
                "assigned_ltv_pct"]
    result[out_cols].sort_values("liquidity_percentile", ascending=False).to_csv(
        "data/csv/ltv_assignment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("data/csv/ltv_tier_summary.csv", index=False, encoding="utf-8-sig")
    print("[저장] data/csv/ltv_assignment.csv, data/csv/ltv_tier_summary.csv")

    plot_result(result)


if __name__ == "__main__":
    main()
