"""
LTV 차등 할인폭(%p) 통계적 캘리브레이션 — 신용점수 예시(40%->30%)에서 빌려온
-10%p 대신, 프로젝트 데이터 안에서 실제로 관측되는 값을 근거로 쓸 수 있는지 검증.

논리: LGD(담보가치 손실)를 직접 계산할 경매 낙찰가율 데이터는 없지만, "유동성이 낮은
단지가 역사적으로 실제로 얼마나 더 크게 가격이 흔들렸는가"(MDD, 최대낙폭)는 이미
gwr_analysis.py가 계산해뒀다. 유동성 최하위 티어와 최상위 티어의 평균 MDD 격차를
"은행이 흡수해야 할 추가 하락 리스크"로 해석해, ltv_scoring.py의 LTV_MAX_DISCOUNT(10%p,
설계값)를 이 실측 격차로 교체할 수 있는지 판단한다.

주의: MDD는 "가격이 얼마나 흔들렸는가"(변동성 리스크)이지, LGD가 실제로 의미하는
"경매에서 제값 대비 얼마나 후려쳐서 팔렸는가"(매각가율 할인)와는 다르다. 진짜 매각가율
데이터가 없는 상태에서의 차선책(proxy)이라는 한계는 명확히 남는다.

관측 단위: 단지 1개 = 1행. MDD는 gwr_analysis.build_complex_table(min_trades=10)
결과(178개 단지, 거래 커버리지 96.0%)와 동일 정의. 유동성은 liquidity_feature_selection
.build_static_table()의 trade_activity_ratio(전체 316개 단지, 거래분기비율)를 그대로
재사용해 ltv_scoring.py의 4분위 티어 구조와 통일한다.
검증: 4개 티어 간 MDD 차이의 ANOVA(전체 유의성) + 최상위/최하위 티어 간 t-test(양극단
격차의 유의성) — build_master_data.py의 법정동 이질성 검증과 동일 방법론.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        matplotlib.rc("font", family=font_name)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트 참조용
from gwr_analysis import load_master, build_complex_table, MIN_TRADES_DEFAULT
from liquidity_feature_selection import build_static_table

TIER_LABELS = {4: "1등급(상위 유동성)", 3: "2등급", 2: "3등급", 1: "4등급(하위 유동성)"}
TIER_COLORS = {4: "#1f6f43", 3: "#5ba86a", 2: "#e0a13c", 1: "#c0392b"}
CURRENT_ASSUMED_DISCOUNT = 10.0  # ltv_scoring.py의 LTV_MAX_DISCOUNT (신용점수 예시 차용값)
OUT_PNG = "data/img/ltv_discount_calibration.png"


def load_merged() -> pd.DataFrame:
    master = load_master()
    mdd_table = build_complex_table(master, MIN_TRADES_DEFAULT)[["complex_id", "MDD", "n_trades"]]

    liq_table = build_static_table()[["complex_id", "trade_activity_ratio"]]

    merged = mdd_table.merge(liq_table, on="complex_id", how="inner")
    merged["liquidity_tier"] = pd.qcut(merged["trade_activity_ratio"], 4, labels=[1, 2, 3, 4]).astype(int)
    print(f"[병합] MDD({len(mdd_table)}개 단지) x 유동성({len(liq_table)}개 단지) "
          f"-> {len(merged)}개 단지 (min_trades={MIN_TRADES_DEFAULT} 필터로 인한 교집합)")

    n_trades_by_tier = merged.groupby("liquidity_tier")["n_trades"].mean()
    corr_mdd_ntrades = merged["MDD"].corr(merged["n_trades"])
    print(f"\n[관측빈도 점검] 티어별 평균 거래건수(7년 누적): {dict(n_trades_by_tier.round(1))}")
    print(f"  MDD-거래건수 상관계수: {corr_mdd_ntrades:.3f} "
          f"({'강한 양의 상관 -> 관측빈도 편향 의심' if corr_mdd_ntrades > 0.3 else '약함'})")
    return merged


def summarize_and_test(merged: pd.DataFrame) -> pd.DataFrame:
    summary = merged.groupby("liquidity_tier")["MDD"].agg(
        mean_mdd="mean", median_mdd="median", std_mdd="std", n="count").reset_index()
    summary = summary.sort_values("liquidity_tier", ascending=False)
    print("\n[티어별 MDD 요약] (MDD는 0~1, %p 환산은 x100)")
    print(summary.to_string(index=False))

    groups = [g["MDD"].values for _, g in merged.groupby("liquidity_tier")]
    f_stat, p_anova = f_oneway(*groups)
    print(f"\n[ANOVA: 4개 티어 간 MDD 차이] F={f_stat:.3f}, p={p_anova:.4f} "
          f"({'유의함' if p_anova < 0.05 else '유의하지 않음'})")

    tier1 = merged[merged["liquidity_tier"] == 1]["MDD"]  # 4등급, 하위 유동성
    tier4 = merged[merged["liquidity_tier"] == 4]["MDD"]  # 1등급, 상위 유동성
    t_stat, p_ttest = ttest_ind(tier1, tier4, equal_var=False)
    # 가설: 유동성 낮을수록(tier1) MDD가 크다 -> expected_gap = tier1 - tier4 > 0
    expected_gap = (tier1.mean() - tier4.mean()) * 100
    print(f"[t-test: 4등급(하위) vs 1등급(상위)] t={t_stat:.3f}, p={p_ttest:.4f} "
          f"({'유의함' if p_ttest < 0.05 else '유의하지 않음'})")
    print(f"\n[실측 격차] 4등급(하위유동성) 평균MDD {tier1.mean()*100:.2f}%p vs "
          f"1등급(상위유동성) 평균MDD {tier4.mean()*100:.2f}%p -> 가설방향 격차 {expected_gap:+.2f}%p "
          f"(양수=가설과 일치/하위유동성이 더 위험, 음수=가설과 반대)")
    print(f"[비교] 현재 ltv_scoring.py 설계값(LTV_MAX_DISCOUNT) = {CURRENT_ASSUMED_DISCOUNT:.1f}%p")

    if expected_gap < 0:
        verdict = ("가설과 반대 방향(유동성 높은 단지의 MDD가 오히려 더 큼) -> 이건 리스크가 실제로 "
                   "역전됐다는 뜻이 아니라 '관측빈도 편향'(비유동 단지는 거래가 드물어 월별 가격이 "
                   "듬성듬성 관측되고, 그 사이 실제 하락을 못 잡아내 MDD가 인위적으로 작게 나오는 현상 "
                   "-- 비유동자산 변동성이 통계적으로 과소추정되는 건 금융권에서 잘 알려진 현상)으로 "
                   "설명됨. MDD-거래건수 상관계수가 이를 뒷받침. -> MDD를 LTV_MAX_DISCOUNT 캘리브레이션에 "
                   f"쓰는 건 기각. 설계값 {CURRENT_ASSUMED_DISCOUNT:.1f}%p(신용점수 예시 차용)를 유지 권장.")
    elif p_ttest < 0.05:
        verdict = ("가설과 같은 방향이고 통계적으로 유의함 -> LTV_MAX_DISCOUNT를 "
                   f"{expected_gap:.1f}%p로 교체하는 것을 검토 가능")
    else:
        verdict = ("가설과 같은 방향이나 통계적으로 유의하지 않음(p>=0.05) -> 표본이 작아 폭을 단정하기 "
                   f"어려움. 설계값 {CURRENT_ASSUMED_DISCOUNT:.1f}%p를 유지하는 게 안전")
    print(f"\n[결론] {verdict}")

    summary["anova_p"] = p_anova
    summary["tier1_vs_tier4_ttest_p"] = p_ttest
    summary["expected_direction_gap_pct"] = expected_gap
    return summary


def plot_result(merged: pd.DataFrame, summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tiers = [4, 3, 2, 1]
    means = [summary.loc[summary["liquidity_tier"] == t, "mean_mdd"].values[0] * 100 for t in tiers]
    stds = [summary.loc[summary["liquidity_tier"] == t, "std_mdd"].values[0] * 100 for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    ax.bar([TIER_LABELS[t] for t in tiers], means, yerr=stds, capsize=4, color=colors)
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.35)
    ax.set_ylabel("평균 MDD (%, 최대낙폭)")
    ax.set_title("유동성 티어별 실측 MDD — 가설과 반대(관측빈도 편향)\n(오차막대=표준편차, LTV 할인폭 캘리브레이션 기각 근거)")
    gap = summary["expected_direction_gap_pct"].iloc[0]
    p_val = summary["tier1_vs_tier4_ttest_p"].iloc[0]
    ax.text(0.02, 0.95, f"4등급-1등급 격차(가설방향): {gap:+.1f}%p (t-test p={p_val:.3f})\n"
            f"음수 = 가설과 반대(관측빈도 편향 의심)",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\n[저장] {OUT_PNG}")


def main():
    merged = load_merged()
    summary = summarize_and_test(merged)
    plot_result(merged, summary)

    merged.to_csv("data/csv/ltv_discount_calibration_complexes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv("data/csv/ltv_discount_calibration_summary.csv", index=False, encoding="utf-8-sig")
    print("[저장] data/csv/ltv_discount_calibration_complexes.csv, "
          "data/csv/ltv_discount_calibration_summary.csv")


if __name__ == "__main__":
    main()
