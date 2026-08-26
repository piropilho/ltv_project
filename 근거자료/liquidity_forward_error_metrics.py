"""
유동성 예측 모델 — 오차지표 / 나이브 베이스라인 비교 / 등급 정확도
=====================================================================
liquidity_forward_prediction.py는 R2(설명력)만 보고했다. R2는 스케일이 없는 상대
지표라 "실제로 얼마나 틀리는지", "그냥 최근 추세를 그대로 반복하는 것보다 얼마나
나은지", "ltv_scoring.py가 실제로 쓰는 4분위 등급을 얼마나 정확히 맞히는지"는 알 수
없다. 이 스크립트는 STEP6 최종 피처셋(R2=0.687)에 대해 같은 walk-forward 폴드
구조에서 세 가지를 추가로 계산한다.

1. MAE/RMSE (원 단위 = 활동비율, 0~1): XGBoost가 reg:squarederror로 학습되므로
   RMSE는 실제 학습 손실을 원 단위로 보여주는 것과 사실상 같다.
2. 나이브 베이스라인 대비 개선폭: "향후 1년도 트레일링 활동비율(최근 4분기 실적)과
   같을 것"이라는 가장 단순한 예측(trailing_activity_ratio를 그대로 예측값으로 사용)
   대비, XGBoost 모델이 실제로 얼마나 더 나은지 검증한다. 모델이 이 베이스라인보다
   못하면 모델의 복잡성이 값어치를 못 한다는 뜻.
3. 등급(4분위 티어) 혼동행렬: ltv_scoring.py가 최종적으로 쓰는 건 raw 예측값이
   아니라 4분위 등급(LTV 40/37/33/30% 매핑 단위)이므로, "등급을 얼마나 정확히
   맞히는가"가 raw R2보다 실제 비즈니스 지표에 더 가깝다. 등급 경계는 각 폴드의
   학습 데이터 분포로 정해(미래 정보 누수 방지) 예측/실제 값을 같은 경계로 나눠
   비교한다. 티어 번호는 ltv_scoring.py 내부 표기와 동일하게 4=상위 유동성.

주의: forward_activity_ratio_1y는 향후 4분기 중 거래분기 비율이라 {0, 0.25, 0.5,
0.75, 1.0} 5개 값만 가능한 이산적인 타겟이다 (학습 데이터의 25%가 정확히 0.0에
몰려있음). 그래서 학습폴드 기준 분위 경계가 겹칠 수 있는데, 이 경우도 그대로
보고한다 -- 인위적으로 매끄럽게 만들지 않음.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))            # 같은 폴더(근거자료) 참조용
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # 프로젝트 루트 참조용
from liquidity_forward_prediction import N_FOLDS, XGB_PARAMS
from liquidity_forward_overfitting_check import build_step6_panel, TARGET_COL

NAIVE_COL = "trailing_activity_ratio"  # 나이브 베이스라인: "최근 4분기 실적이 향후 1년에도 이어질 것"
OUT_PNG = "data/img/liquidity_forward_error_metrics.png"


def evaluate_folds(panel: pd.DataFrame, feature_cols: list, target_col: str):
    t_values = sorted(panel["t_idx"].unique())
    warmup = max(1, int(len(t_values) * 0.5))
    test_t_values = t_values[warmup:]
    fold_edges = np.array_split(test_t_values, min(N_FOLDS, len(test_t_values)))

    fold_rows = []
    actual_tiers_all, pred_tiers_all = [], []

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
        actual = test[target_col].to_numpy()
        naive = test[NAIVE_COL].to_numpy()

        mae_model = np.mean(np.abs(pred - actual))
        rmse_model = np.sqrt(np.mean((pred - actual) ** 2))
        mae_naive = np.mean(np.abs(naive - actual))
        rmse_naive = np.sqrt(np.mean((naive - actual) ** 2))

        # 등급 경계: 학습 데이터의 타겟 분위 (테스트 정보 누수 방지)
        # right=True: 경계값과 정확히 같은 값은 하위 등급으로 포함 (타겟이 이산적이라
        # 25%가 정확히 0.0에 몰려있음 -- right=False면 0.0이 전부 2등급으로 밀려
        # 1등급이 항상 텅 비는 아티팩트가 생김)
        edges = train[target_col].quantile([0.25, 0.5, 0.75]).to_numpy()
        n_unique_edges = len(np.unique(edges))
        pred_tier = np.digitize(pred, edges, right=True) + 1
        actual_tier = np.digitize(actual, edges, right=True) + 1
        actual_tiers_all.extend(actual_tier.tolist())
        pred_tiers_all.extend(pred_tier.tolist())

        tier_acc = np.mean(pred_tier == actual_tier)
        tier_off_by_2plus = np.mean(np.abs(pred_tier - actual_tier) >= 2)

        fold_rows.append({
            "fold": fold_i, "n_test": len(test),
            "mae_model": mae_model, "rmse_model": rmse_model,
            "mae_naive": mae_naive, "rmse_naive": rmse_naive,
            "mae_improve_pct": (mae_naive - mae_model) / mae_naive * 100 if mae_naive else np.nan,
            "rmse_improve_pct": (rmse_naive - rmse_model) / rmse_naive * 100 if rmse_naive else np.nan,
            "tier_accuracy": tier_acc, "tier_off_by_2plus_pct": tier_off_by_2plus * 100,
            "n_unique_tier_edges": n_unique_edges,
        })

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, np.array(actual_tiers_all), np.array(pred_tiers_all)


def print_summary(fold_df: pd.DataFrame):
    print("[폴드별 오차지표 / 나이브 베이스라인 비교 / 등급 정확도]")
    print(fold_df.round(4).to_string(index=False))

    w = fold_df["n_test"]
    def wavg(col):
        return float(np.average(fold_df[col], weights=w))

    print(f"\n[가중평균] MAE(모델)={wavg('mae_model'):.4f}  MAE(나이브)={wavg('mae_naive'):.4f}  "
          f"개선율={wavg('mae_improve_pct'):+.1f}%")
    print(f"[가중평균] RMSE(모델)={wavg('rmse_model'):.4f}  RMSE(나이브)={wavg('rmse_naive'):.4f}  "
          f"개선율={wavg('rmse_improve_pct'):+.1f}%")
    print(f"[가중평균] 등급 정확도={wavg('tier_accuracy'):.1%}  2등급 이상 크게 틀린 비율="
          f"{wavg('tier_off_by_2plus_pct'):.1f}%")

    if (fold_df["n_unique_tier_edges"] < 3).any():
        print("\n[참고] 일부 폴드는 학습 타겟 분포가 이산적(특히 0.0에 몰림)이라 4분위 경계 중 "
              "일부가 겹쳤음 (n_unique_tier_edges<3) -- 그 폴드는 사실상 3개 구간으로 나뉜 것.")

    return {"weighted_mae_model": wavg("mae_model"), "weighted_mae_naive": wavg("mae_naive"),
            "weighted_mae_improve_pct": wavg("mae_improve_pct"),
            "weighted_rmse_model": wavg("rmse_model"), "weighted_rmse_naive": wavg("rmse_naive"),
            "weighted_rmse_improve_pct": wavg("rmse_improve_pct"),
            "weighted_tier_accuracy": wavg("tier_accuracy")}


def plot_results(fold_df: pd.DataFrame, actual_tiers: np.ndarray, pred_tiers: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    x = fold_df["fold"]
    width = 0.35
    ax.bar(x - width / 2, fold_df["mae_model"], width, label="모델(XGBoost) MAE", color="#1f4e8c")
    ax.bar(x + width / 2, fold_df["mae_naive"], width, label="나이브 베이스라인 MAE", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xlabel("폴드 (뒤로 갈수록 학습표본 증가)")
    ax.set_ylabel("MAE (활동비율, 0~1)")
    ax.set_title("모델 vs 나이브 베이스라인(직전 추세 반복)")
    ax.legend(fontsize=8)

    ax = axes[1]
    tiers = [1, 2, 3, 4]
    cm = pd.crosstab(pd.Series(actual_tiers, name="실제 등급"),
                      pd.Series(pred_tiers, name="예측 등급")).reindex(index=tiers, columns=tiers, fill_value=0)
    cm_norm = cm.div(cm.sum(axis=1), axis=0)  # 실제 등급별 행 정규화(재현율 관점)
    im = ax.imshow(cm_norm.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm.values[i, j]}", ha="center", va="center",
                    color="white" if cm_norm.values[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(tiers)
    ax.set_yticks(range(4)); ax.set_yticklabels(tiers)
    ax.set_xlabel("예측 등급 (4=상위 유동성)")
    ax.set_ylabel("실제 등급 (4=상위 유동성)")
    ax.set_title("등급 혼동행렬 (숫자=건수, 색=행 기준 비율)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("유동성 예측 모델 — 오차지표 / 등급 정확도 점검", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\n[저장] {OUT_PNG}")
    return cm


def main():
    panel, feature_cols = build_step6_panel()
    fold_df, actual_tiers, pred_tiers = evaluate_folds(panel, feature_cols, TARGET_COL)
    summary = print_summary(fold_df)
    cm = plot_results(fold_df, actual_tiers, pred_tiers)

    fold_df.to_csv("data/csv/liquidity_forward_error_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv("data/csv/liquidity_forward_error_metrics_summary.csv",
                                    index=False, encoding="utf-8-sig")
    cm.to_csv("data/csv/liquidity_forward_tier_confusion_matrix.csv", encoding="utf-8-sig")
    print("[저장] data/csv/liquidity_forward_error_metrics.csv, "
          "data/csv/liquidity_forward_error_metrics_summary.csv, "
          "data/csv/liquidity_forward_tier_confusion_matrix.csv")


if __name__ == "__main__":
    main()
