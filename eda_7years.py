"""
동대문구 아파트 실거래가 EDA (2019~2025, 7개년)
=======================================================
입력: 19_25_trading.csv (7개년 병합·클리닝·지오코딩 완료된 마스터 데이터)
출력:
  - master_cleaned_19_25.csv        클리닝 완료 마스터 데이터
  - eda_summary/yearly_volume.csv    연도별 거래량/해제율/월커버리지
  - eda_summary/skewness_by_year.csv 연도별 왜도(원값/로그변환)
  - eda_summary/correlation.csv      전체기간 상관행렬
  - eda_summary/regime_summary.csv   국면별 CV/MDD/거래량 요약
  - eda_summary/dong_yearly.csv      법정동x연도 평균단가 피벗
  - eda_summary/threshold_table.csv  단지별 거래건수 임계값 커버리지
  - eda_summary/monthly_trend.png    월별 가격·거래량 추이 (정책이벤트 표시)
"""

import os
import math

import numpy as np
import pandas as pd
from scipy.stats import skew, f_oneway
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

# 한글 폰트 설정 (맥: AppleGothic, 리눅스: NanumGothic, 윈도우: Malgun Gothic)
for font_path in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        matplotlib.rc("font", family=fm.FontProperties(fname=font_path).get_name())
        break
else:
    for font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            matplotlib.rc("font", family=font_name)
            break
matplotlib.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------
SRC_CSV = "data/19_25_trading.csv"
OUT_DIR = "data"
MASTER_OUT = "data/master_19_25_cleaning.csv"

# 정책 이벤트 (LTV 강화 계열). 필요시 팀 결정에 따라 추가/수정.
POLICY_EVENTS = {
    "6.17대책(2020)": "2020-06-17",
    "7.10대책(2020)": "2020-07-10",
    "6.27대책(2025)": "2025-06-27",
    "9.7대책(2025)": "2025-09-07",
    "10.15대책(2025)": "2025-10-15",
}

# 국면 구분 (6.17/6.27/9.7/10.15는 정책 발표일로 고정, 상승기/급락기/회복기 경계는
# monthly mean price_per_m2의 실제 변곡점으로 재산정: 고점 2021-09, 저점 2022-10)
REGIME_BOUNDARIES = [
    ("2019~2020상반기(baseline)", "2019-01-01", "2020-06-16"),
    ("2020.6.17~2021.9(상승기)", "2020-06-17", "2021-09-30"),
    ("2021.10~2022.10(급락기)", "2021-10-01", "2022-10-31"),
    ("2022.11~2025.6(회복기)", "2022-11-01", "2025-06-26"),
    ("2025.6.27~10.15(전환기)", "2025-06-27", "2025-10-14"),
    ("2025.10.15~(신규제 정착기)", "2025-10-15", "2026-12-31"),
]


# ------------------------------------------------------------------
# 1. 데이터 로드 + 클리닝
# ------------------------------------------------------------------
def load_and_merge_raw() -> pd.DataFrame:
    if not os.path.exists(SRC_CSV):
        raise FileNotFoundError(f"{SRC_CSV} 파일을 찾을 수 없음")
    raw = pd.read_csv(SRC_CSV, encoding="utf-8-sig")
    print(f"[로드] {SRC_CSV} -> {raw.shape}")
    return raw


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["dealAmount"] = df["dealAmount"].astype(str).str.replace(",", "", regex=False).astype(float)

    n_before = len(df)
    df = df[df["cdealType"].isna()].copy()
    n_after_cancel = len(df)
    df = df.drop_duplicates()
    n_after_dup = len(df)

    df["price_per_m2"] = df["dealAmount"] / df["excluUseAr"]
    df["building_age"] = df["dealYear"] - df["buildYear"]
    df["contract_date"] = pd.to_datetime(
        df["dealYear"].astype(str) + "-"
        + df["dealMonth"].astype(str).str.zfill(2) + "-"
        + df["dealDay"].astype(str).str.zfill(2)
    )
    df["year_month"] = df["contract_date"].dt.to_period("M")
    df["complex_id"] = df.groupby(["lat", "lon"]).ngroup()

    print(f"[클리닝] 원본 {n_before} -> 해제거래 제외 {n_after_cancel} -> 중복제거 {n_after_dup}")
    print(f"[클리닝] 고유 단지 수: {df['complex_id'].nunique()}개, "
          f"기간: {df['contract_date'].min().date()} ~ {df['contract_date'].max().date()}")
    return df


# ------------------------------------------------------------------
# 2. 연도별 데이터 볼륨 점검
# ------------------------------------------------------------------
def check_yearly_volume(raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(raw["dealYear"].unique()):
        raw_y = raw[raw["dealYear"] == y]
        clean_y = df_clean[df_clean["dealYear"] == y]
        n_cancel = raw_y["cdealType"].notna().sum()
        months = sorted(clean_y["dealMonth"].unique())
        rows.append({
            "year": y,
            "raw_count": len(raw_y),
            "cancel_count": n_cancel,
            "cancel_pct": round(n_cancel / len(raw_y) * 100, 1) if len(raw_y) else 0,
            "clean_count": len(clean_y),
            "n_months_covered": len(months),
        })
    result = pd.DataFrame(rows)
    print("\n[연도별 볼륨]")
    print(result.to_string(index=False))
    return result


# ------------------------------------------------------------------
# 3. 단변량 분포 (연도별 왜도)
# ------------------------------------------------------------------
def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(df["dealYear"].unique()):
        sub = df[df["dealYear"] == y]["price_per_m2"]
        raw_skew = skew(sub)
        log_skew = skew(sub.apply(math.log))
        rows.append({"year": y, "raw_skew": round(raw_skew, 3), "log_skew": round(log_skew, 3)})
    result = pd.DataFrame(rows)
    print("\n[연도별 왜도]")
    print(result.to_string(index=False))
    return result


# ------------------------------------------------------------------
# 4. 이변량 / 다중공선성
# ------------------------------------------------------------------
def bivariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["dealAmount", "excluUseAr", "floor", "building_age", "price_per_m2"]
    corr = df[cols].corr().round(3)
    print("\n[상관행렬]")
    print(corr)
    return corr


# ------------------------------------------------------------------
# 5. 월별 트렌드 시각화 (정책이벤트 표시)
# ------------------------------------------------------------------
def plot_monthly_trend(df: pd.DataFrame, out_path: str):
    monthly = df.groupby("year_month").agg(
        n_trades=("price_per_m2", "count"),
        mean_price=("price_per_m2", "mean"),
    ).reset_index()
    monthly["date"] = monthly["year_month"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(monthly["date"], monthly["mean_price"], color="#2a78d6", linewidth=2)
    axes[0].set_ylabel("평균 ㎡당가격(만원)")
    axes[0].set_title("동대문구 아파트 월별 가격·거래량 추이 (2019~2025)")

    axes[1].bar(monthly["date"], monthly["n_trades"], width=20, color="#2a78d6")
    axes[1].set_ylabel("거래건수")

    for ax in axes:
        for label, date_str in POLICY_EVENTS.items():
            ax.axvline(pd.to_datetime(date_str), color="#e34948", linestyle="--", linewidth=1, alpha=0.7)

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[저장] 월별 트렌드 차트 -> {out_path}")


# ------------------------------------------------------------------
# 6. 국면별 요약 (CV, MDD, 거래량)
# ------------------------------------------------------------------
def compute_mdd(price_series: pd.Series) -> float:
    """시계열의 최대 낙폭(고점 대비 최대 하락률, 0~1)"""
    if len(price_series) < 2:
        return np.nan
    cummax = price_series.cummax()
    drawdown = (price_series - cummax) / cummax
    return drawdown.min() * -1  # 양수로 표현 (예: 0.15 = 15% 하락)


def regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in REGIME_BOUNDARIES:
        mask = (df["contract_date"] >= start) & (df["contract_date"] <= end)
        sub = df[mask]
        if sub.empty:
            continue
        monthly_price = sub.groupby("year_month")["price_per_m2"].mean().sort_index()
        cv_cross = sub["price_per_m2"].std() / sub["price_per_m2"].mean()
        cv_timeseries = monthly_price.std() / monthly_price.mean()
        mdd = compute_mdd(monthly_price)
        rows.append({
            "regime": label,
            "period": f"{start} ~ {end}",
            "n_trades": len(sub),
            "n_months": len(monthly_price),
            "mean_price": round(sub["price_per_m2"].mean(), 1),
            "CV_cross": round(cv_cross, 4),
            "CV_timeseries": round(cv_timeseries, 4),
            "MDD": round(mdd, 4) if not pd.isna(mdd) else None,
        })
    result = pd.DataFrame(rows)
    print("\n[국면별 요약: CV(횡단면/시계열)/MDD]")
    print(result.to_string(index=False))
    return result


# ------------------------------------------------------------------
# 7. 공간(법정동) 이질성
# ------------------------------------------------------------------
def spatial_heterogeneity(df: pd.DataFrame):
    pivot = df.groupby(["dealYear", "umdNm"])["price_per_m2"].mean().unstack().round(0)
    print("\n[법정동x연도 평균 ㎡당가격]")
    print(pivot)

    groups = [g["price_per_m2"].values for _, g in df.groupby("umdNm")]
    f_stat, p_val = f_oneway(*groups)
    print(f"\n[ANOVA] F={f_stat:.2f}, p={p_val:.6f} "
          f"({'유의함' if p_val < 0.05 else '유의하지 않음'})")
    return pivot, (f_stat, p_val)


# ------------------------------------------------------------------
# 8. 단지별 거래건수 임계값 커버리지
# ------------------------------------------------------------------
def trade_threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("complex_id").size().sort_values(ascending=False)
    total_c, total_t = len(counts), len(df)
    rows = []
    for th in [1, 3, 5, 10, 20, 30, 50]:
        kc = (counts >= th).sum()
        kt = counts[counts >= th].sum()
        rows.append({
            "min_trades": th,
            "n_complex": kc,
            "complex_pct": round(kc / total_c * 100, 1),
            "n_trades_covered": int(kt),
            "trade_pct": round(kt / total_t * 100, 1),
        })
    result = pd.DataFrame(rows)
    print("\n[거래건수 임계값 커버리지]")
    print(result.to_string(index=False))
    return result


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = load_and_merge_raw()
    df = clean_data(raw)
    df.to_csv(MASTER_OUT, index=False, encoding="utf-8-sig")

    yearly_volume = check_yearly_volume(raw, df)
    yearly_volume.to_csv(f"{OUT_DIR}/yearly_volume.csv", index=False, encoding="utf-8-sig")

    skewness = univariate_analysis(df)
    skewness.to_csv(f"{OUT_DIR}/skewness_by_year.csv", index=False, encoding="utf-8-sig")

    corr = bivariate_analysis(df)
    corr.to_csv(f"{OUT_DIR}/correlation.csv", encoding="utf-8-sig")

    plot_monthly_trend(df, f"{OUT_DIR}/monthly_trend.png")

    regime = regime_summary(df)
    regime.to_csv(f"{OUT_DIR}/regime_summary.csv", index=False, encoding="utf-8-sig")

    dong_pivot, anova_result = spatial_heterogeneity(df)
    dong_pivot.to_csv(f"{OUT_DIR}/dong_yearly.csv", encoding="utf-8-sig")

    threshold_table = trade_threshold_table(df)
    threshold_table.to_csv(f"{OUT_DIR}/threshold_table.csv", index=False, encoding="utf-8-sig")

    print(f"\n모든 요약 결과 저장 완료: {OUT_DIR}/")


if __name__ == "__main__":
    main()
