"""
경사도 반경(100/150/200/300m) 선택 결과 시각화
=================================================
입력: data/csv/gwr_slope_radius_comparison_{MDD,CV}.csv (gwr_feature_selection.py 산출물 —
      반경별 중첩 K-fold out-of-sample R²)

목적: "100m이 가장 적합하다"는 결론을 숫자표가 아니라 그래프로 직관적으로 보여주고,
      반경이 멀어질수록 설명력이 얼마나/어떻게 떨어지는지 한눈에 비교할 수 있게 한다.

스케일 처리: R² 값 자체가 0.01~0.04 수준으로 작아서, y축을 0~1 같은 일반적인 범위로 두면
      네 반경 차이가 안 보인다. 그래서 실제 데이터 범위에 맞춰 y축을 좁게 잡고, 각 지점에
      "100m 대비 상대적으로 몇 % 수준인지"를 같이 표시해 스케일 왜곡 없이 비교 가능하게 한다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

for font_name in ["AppleGothic", "Malgun Gothic", "NanumGothic"]:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        matplotlib.rc("font", family=font_name)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

MDD_CSV = "data/csv/gwr_slope_radius_comparison_MDD.csv"
CV_CSV = "data/csv/gwr_slope_radius_comparison_CV.csv"
OUT_PNG = "data/img/radius_selection.png"

Y_AXIS_PADDING_RATIO = 0.25  # 데이터 범위 대비 상하 여백 비율 (너무 빡빡하게 붙지 않도록)


def plot_panel(ax, df: pd.DataFrame, target_label: str):
    df = df.sort_values("radius_m").reset_index(drop=True)
    best_idx = df["nested_cv_r2"].idxmax()
    best_radius = df.loc[best_idx, "radius_m"]
    best_r2 = df.loc[best_idx, "nested_cv_r2"]

    colors = ["#c0392b" if r == best_radius else "#1f4e8c" for r in df["radius_m"]]

    ax.plot(df["radius_m"], df["nested_cv_r2"], color="#9aa5b1", linewidth=1.5, zorder=1)
    ax.scatter(df["radius_m"], df["nested_cv_r2"], c=colors, s=110, zorder=2, edgecolor="white")
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1, alpha=0.7)

    y_min, y_max = df["nested_cv_r2"].min(), df["nested_cv_r2"].max()
    span = max(y_max - y_min, 1e-6)
    pad = span * Y_AXIS_PADDING_RATIO
    ax.set_ylim(y_min - pad, y_max + pad)

    all_negative = (df["nested_cv_r2"] < 0).all()
    for _, row in df.iterrows():
        r2 = row["nested_cv_r2"]
        if row["radius_m"] == best_radius:
            note = f"{r2:.4f}\n(최적)"
        elif all_negative:
            # R2가 전부 음수(평균 예측보다 못함)인 경우, "최적 대비 %"는 부호 때문에
            # 오히려 왜곡돼 보이므로(더 나쁜 값이 %가 더 커 보임) 0 대비 절대 차이로 표기
            note = f"{r2:.4f}\n(0 대비 {r2:+.4f})"
        else:
            pct_of_best = r2 / best_r2 * 100
            note = f"{r2:.4f}\n(최적 대비 {pct_of_best:.0f}%)"
        va = "bottom" if r2 >= (y_min + y_max) / 2 else "top"
        offset = span * 0.08 if va == "bottom" else -span * 0.08
        ax.annotate(note, (row["radius_m"], r2 + offset), ha="center", va=va, fontsize=9)

    ax.set_xticks(df["radius_m"])
    ax.set_xlabel("경사도 반경 (m)")
    ax.set_ylabel("중첩 K-fold out-of-sample R²")
    subtitle = f"(빨간 점 = 최적 반경 {best_radius}m)"
    if all_negative:
        subtitle += "\n※ 전 구간 R²<0 — 어느 반경도 평균 예측보다 못함"
    ax.set_title(f"{target_label} — 반경별 예측력 비교\n{subtitle}")


def main():
    mdd_df = pd.read_csv(MDD_CSV)
    cv_df = pd.read_csv(CV_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_panel(axes[0], mdd_df, "MDD")
    plot_panel(axes[1], cv_df, "CV")

    fig.suptitle("경사도 반경 선택 — out-of-sample R² 비교 (gwr_feature_selection.py 결과)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"[저장] -> {OUT_PNG}")

    for label, df in [("MDD", mdd_df), ("CV", cv_df)]:
        df = df.sort_values("radius_m")
        best = df.loc[df["nested_cv_r2"].idxmax()]
        print(f"\n[{label}] 반경별 R² (100m 대비 변화):")
        for _, row in df.iterrows():
            delta = row["nested_cv_r2"] - best["nested_cv_r2"]
            print(f"  {int(row['radius_m']):3d}m: R2={row['nested_cv_r2']:+.4f}  "
                  f"(최적 대비 {delta:+.4f})")


if __name__ == "__main__":
    main()
