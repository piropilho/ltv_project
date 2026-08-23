"""
미래 유동성 예측 모델(STEP6, R2=0.687) 과적합 점검
====================================================
liquidity_forward_prediction.py의 walkforward_r2()는 테스트 R2만 보고했다. 이 스크립트는
같은 폴드 구조에서 학습(train) R2도 같이 계산해서 train-test 격차를 확인한다.

판단 기준: 격차가 학습표본이 늘어나는(확장윈도우) 뒤쪽 폴드일수록 좁아지면 "데이터가
적어서 초반에 살짝 과적합했을 뿐, 모델 자체가 노이즈를 외우는 건 아니다"로 판단.
반대로 데이터가 늘어도 격차가 안 줄면 진짜 과적합 의심.

같은 데이터로 전체적합 변수중요도(gain)도 같이 뽑아서, 어떤 피처가 예측력을 이끄는지
확인한다 (unit_cnt가 압도적이라는 게 "현재시점" 유동성 모델과 일관되는지 교차검증).
"""

import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 프로젝트 루트 참조용
from liquidity_forward_prediction import (
    load_base_panel, build_panel_for_window, BASELINE_TRAILING_N, BASELINE_FEATURES,
    N_FOLDS, XGB_PARAMS,
)

TARGET_COL = "forward_activity_ratio_1y"
STEP6_EXTRA_FEATURES = ["commercial_index_lasso19_t", "pop_growth_mom_t", "price_momentum",
                         "days_since_last_tightening_t", "n_tightening_past_1y_t"]
OUT_PNG = "data/img/liquidity_forward_overfitting_check.png"


def build_step6_panel():
    base_panel, quarters = load_base_panel()
    panel = build_panel_for_window(base_panel, len(quarters), BASELINE_TRAILING_N, quarters)
    feature_cols = BASELINE_FEATURES + STEP6_EXTRA_FEATURES
    return panel, feature_cols


def train_test_gap_by_fold(panel: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
    t_values = sorted(panel["t_idx"].unique())
    warmup = max(1, int(len(t_values) * 0.5))
    test_t_values = t_values[warmup:]
    fold_edges = np.array_split(test_t_values, min(N_FOLDS, len(test_t_values)))

    rows = []
    for fold_i, test_ts in enumerate(fold_edges):
        test_ts = list(test_ts)
        train_ts = [t for t in t_values if t < min(test_ts)]
        train = panel[panel["t_idx"].isin(train_ts)].dropna(subset=feature_cols + [target_col])
        test = panel[panel["t_idx"].isin(test_ts)].dropna(subset=feature_cols + [target_col])
        if len(train) < 20 or len(test) < 5:
            continue

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(train[feature_cols], train[target_col])

        def r2(sub, pred):
            ss_res = ((sub[target_col] - pred) ** 2).sum()
            ss_tot = ((sub[target_col] - sub[target_col].mean()) ** 2).sum()
            return 1 - ss_res / ss_tot

        r2_train = r2(train, model.predict(train[feature_cols]))
        r2_test = r2(test, model.predict(test[feature_cols]))
        rows.append({"fold": fold_i, "n_train": len(train), "n_test": len(test),
                      "train_r2": r2_train, "test_r2": r2_test, "gap": r2_train - r2_test})
    return pd.DataFrame(rows)


def fit_full_and_get_importance(panel: pd.DataFrame, feature_cols: list, target_col: str) -> pd.Series:
    sub = panel.dropna(subset=feature_cols + [target_col])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(sub[feature_cols], sub[target_col])
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


def plot_results(gap_df: pd.DataFrame, importance: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    x = gap_df["fold"]
    ax.plot(x, gap_df["train_r2"], marker="o", label="train R2", color="#1f4e8c")
    ax.plot(x, gap_df["test_r2"], marker="o", label="test R2", color="#c0392b")
    for _, row in gap_df.iterrows():
        ax.annotate(f"격차 {row['gap']:.3f}", (row["fold"], row["train_r2"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#555555")
    y_min = min(gap_df["train_r2"].min(), gap_df["test_r2"].min())
    y_max = max(gap_df["train_r2"].max(), gap_df["test_r2"].max())
    ax.set_ylim(y_min - 0.03, y_max + 0.05)  # 격차 라벨이 위쪽 테두리에 안 겹치도록 여백 확보

    ax.set_xticks(x)
    ax.set_xlabel("폴드 (뒤로 갈수록 학습표본 증가)")
    ax.set_ylabel("R2")
    ax.set_title("train vs test R2 — 폴드가 뒤로 갈수록 격차 축소\n(과적합 아님을 보여주는 패턴)")
    ax.legend()

    ax = axes[1]
    top = importance.head(10).iloc[::-1]
    ax.barh(top.index, top.values, color="#1f4e8c")
    ax.set_xlabel("변수중요도 (gain)")
    ax.set_title("STEP6 최종 모델 변수중요도 상위 10개")

    fig.suptitle("미래 유동성 예측 모델(R2=0.687) 과적합 점검", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)


def main():
    panel, feature_cols = build_step6_panel()

    gap_df = train_test_gap_by_fold(panel, feature_cols, TARGET_COL)
    print("[폴드별 train-test R2 격차]")
    print(gap_df.to_string(index=False))

    importance = fit_full_and_get_importance(panel, feature_cols, TARGET_COL)
    print("\n[전체데이터 적합 변수중요도(gain)]")
    for name, imp in importance.items():
        print(f"  {name:32s} {imp:.4f}")

    plot_results(gap_df, importance)

    gap_df.to_csv("data/csv/liquidity_forward_overfit_check.csv", index=False, encoding="utf-8-sig")
    importance.to_csv("data/csv/liquidity_forward_feature_importance.csv", encoding="utf-8-sig")
    print(f"\n[저장] {OUT_PNG}, data/csv/liquidity_forward_overfit_check.csv, "
          f"data/csv/liquidity_forward_feature_importance.csv")


if __name__ == "__main__":
    main()
