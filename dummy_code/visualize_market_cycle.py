"""
동대문구 아파트 시장 사이클 시각화
=======================================================
입력: data/master_19_25_cleaning.csv (eda_7years.py 산출물)
출력: data/market_cycle.png

- 월별 평균 ㎡당가격 추이 위에 REGIME_BOUNDARIES 구간을 음영으로 표시
- POLICY_EVENTS(정책 발표일)는 빨간 점선으로 표시
- 데이터로 확인된 실제 고점(2021-09)/저점(2022-10)을 화살표로 강조
- 관측 개월 수가 6개월 미만인 국면은 "관측기간 부족, 판단 보류"를 라벨에 자동으로 붙임
  (2025.10.15~ 구간처럼 아직 사이클로 단정하기엔 이른 구간을 시각적으로 구분하기 위함)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

from eda_7years import REGIME_BOUNDARIES, POLICY_EVENTS

for font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        matplotlib.rc("font", family=font_name)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

SRC_CSV = "data/master_19_25_cleaning.csv"
OUT_PNG = "data/market_cycle.png"

REGIME_COLORS = ["#dbe7f6", "#d7ecd4", "#fbe3cf", "#e3ddf3", "#fbd6d6", "#e6e6e6"]
MIN_MONTHS_RELIABLE = 6


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(SRC_CSV)
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["year_month"] = df["contract_date"].dt.to_period("M")
    monthly = df.groupby("year_month")["price_per_m2"].mean().sort_index()
    monthly.index = monthly.index.to_timestamp()
    return df, monthly


def plot_market_cycle(df: pd.DataFrame, monthly: pd.Series, out_path: str):
    data_start, data_end = monthly.index.min(), monthly.index.max()
    fig, (ax, ax_vol) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ymin, ymax = monthly.min() * 0.92, monthly.max() * 1.18

    # 국면 음영 + 번호 라벨 (좁은 구간에서 텍스트가 겹치지 않도록 번호+범례 방식 사용)
    legend_lines = []
    for i, (label, start, end) in enumerate(REGIME_BOUNDARIES):
        start_ts = max(pd.Timestamp(start), data_start)
        end_ts = min(pd.Timestamp(end), data_end)
        if start_ts >= end_ts:
            continue

        mask = (df["contract_date"] >= start_ts) & (df["contract_date"] <= end_ts)
        n_months = df.loc[mask, "year_month"].nunique()

        color = REGIME_COLORS[i % len(REGIME_COLORS)]
        for a in (ax, ax_vol):
            a.axvspan(start_ts, end_ts, color=color, alpha=0.45, zorder=0)

        tag = f"①②③④⑤⑥⑦⑧"[i]
        mid = start_ts + (end_ts - start_ts) / 2
        ax.text(mid, ymax * 0.985, tag, ha="center", va="top", fontsize=11, fontweight="bold")

        note = f" (관측 {n_months}개월, 판단 보류)" if n_months < MIN_MONTHS_RELIABLE else ""
        legend_lines.append(f"{tag} {label}{note}")

    legend_text = "\n".join(legend_lines)
    ax.text(
        0.99, 0.02, legend_text, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#999999"),
    )

    # 가격 추이
    ax.plot(monthly.index, monthly.values, color="#1f4e8c", linewidth=2, zorder=3)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("평균 ㎡당가격(만원)")
    ax.set_title("동대문구 아파트 시장 사이클 (2019~2025)\n음영=국면 구분, 빨간 점선=정책 발표일")

    # 실제 고점/저점 강조 (2021 상승기 고점, 2022 급락기 저점)
    peak_month = monthly.idxmax()
    ax.annotate(
        f"실제 고점 {peak_month:%Y-%m}\n{monthly.max():.0f}만원",
        xy=(peak_month, monthly[peak_month]), xytext=(-60, 25), textcoords="offset points",
        fontsize=8, color="#1f4e8c", arrowprops=dict(arrowstyle="->", color="#1f4e8c"),
    )
    window_2022 = monthly[(monthly.index >= "2022-01-01") & (monthly.index <= "2023-06-30")]
    if not window_2022.empty:
        trough_month = window_2022.idxmin()
        ax.annotate(
            f"실제 저점 {trough_month:%Y-%m}\n{window_2022.min():.0f}만원",
            xy=(trough_month, monthly[trough_month]), xytext=(20, -35), textcoords="offset points",
            fontsize=8, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"),
        )

    # 거래량
    monthly_n = df.groupby("year_month").size()
    monthly_n.index = monthly_n.index.to_timestamp()
    ax_vol.bar(monthly_n.index, monthly_n.values, width=20, color="#5a7ca8", zorder=3)
    ax_vol.set_ylabel("거래건수")

    # 정책 이벤트 수직선
    for pname, pdate in POLICY_EVENTS.items():
        pts = pd.Timestamp(pdate)
        if data_start <= pts <= data_end:
            for a in (ax, ax_vol):
                a.axvline(pts, color="#d62728", linestyle="--", linewidth=1, alpha=0.8, zorder=2)
            ax_vol.text(pts, -ax_vol.get_ylim()[1] * 0.05, pname, rotation=90,
                        fontsize=7, color="#d62728", va="top", ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[저장] 시장 사이클 차트 -> {out_path}")


def main():
    df, monthly = load_data()
    plot_market_cycle(df, monthly, OUT_PNG)


if __name__ == "__main__":
    main()
