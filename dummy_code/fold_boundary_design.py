"""
walk-forward 검증 폴드 경계 재설계 — 가격 시계열 기반 데이터 주도 전환점 탐지
=======================================================================
배경: 지금까지 train_model.py의 walk-forward 폴드는 달력 기준 균등분할(4폴드)이었는데,
폴드 수가 적어 폴드 하나의 우연한 결과가 가중평균 전체를 좌우하는 문제가 반복 관찰됨
(예: 금리/매매수급동향 피처 실험에서 폴드 하나만 크게 개선되는데도 가중평균 R2만 보면
"개선"으로 잘못 판정될 뻔함).

원칙: 이 스크립트는 어떤 피처(금리, 매매수급동향 등) 테스트 결과도 참고하지 않는다.
      오직 data/master_19_25_cleaning.csv의 월별 평균 가격 시계열만 보고 전환점을 찾는다.
      (특정 피처가 잘 나온 구간을 살리는 방향으로 경계를 조정하는 건 평가 기준을 결과에
      맞춰 사후적으로 바꾸는 것이므로 금지)

방법:
1. 월별 평균 price_per_m2의 전월 대비 방향(부호)을 계산
2. 부호가 바뀌어도 MIN_CONFIRM_MONTHS 개월 이상 새 방향이 유지돼야 "확정된 전환"으로 인정
   (1개월짜리 반짝 반전은 노이즈로 보고 무시)
3. 확정된 전환점들을 경계로 구간(국면) 분할
4. 구간 길이가 MIN_SEGMENT_MONTHS 미만이면 인접 구간에 병합
5. 결과를 기존에 수기로 판단했던 REGIME_BOUNDARIES(eda_7years.py)와 대조해 타당성 검증
"""

import numpy as np
import pandas as pd

MASTER_CSV = "data/master_19_25_cleaning.csv"

MIN_CONFIRM_MONTHS = 2   # 방향 전환이 "확정"되려면 유지돼야 하는 최소 개월 수
MIN_SEGMENT_MONTHS = 4   # 이보다 짧은 구간은 인접 구간에 병합 (학습/검증 최소 표본 확보)

OUT_CSV = "data/fold_boundaries.csv"


def load_monthly_price() -> pd.Series:
    df = pd.read_csv(MASTER_CSV)
    df["contract_date"] = pd.to_datetime(df["contract_date"])
    df["year_month"] = df["contract_date"].dt.to_period("M")
    monthly = df.groupby("year_month")["price_per_m2"].mean().sort_index()
    return monthly


def detect_turning_points(monthly: pd.Series) -> list:
    """
    상태기계 방식: 현재 국면의 부호와 다른 방향이 MIN_CONFIRM_MONTHS개월 연속 관측돼야
    그 구간의 시작월을 새 국면의 경계로 확정한다. 그 전까지는 노이즈로 보고 무시.
    반환: 국면 시작월(Period) 리스트 (첫 달 포함)
    """
    months = monthly.index
    diffs = monthly.diff()
    signs = np.sign(diffs)  # 첫 값은 NaN

    first_nonzero = next(i for i in range(1, len(signs)) if signs.iloc[i] != 0)
    current_sign = signs.iloc[first_nonzero]
    boundaries = [months[0]]

    pending_sign, pending_count, pending_start = None, 0, None
    for i in range(first_nonzero + 1, len(signs)):
        s = signs.iloc[i]
        if s == 0 or s == current_sign:
            pending_sign, pending_count, pending_start = None, 0, None
            continue
        if s == pending_sign:
            pending_count += 1
        else:
            pending_sign, pending_count, pending_start = s, 1, i

        if pending_count >= MIN_CONFIRM_MONTHS:
            boundary_idx = pending_start
            boundaries.append(months[boundary_idx])
            current_sign = pending_sign
            pending_sign, pending_count, pending_start = None, 0, None

    return boundaries


def merge_short_segments(boundaries: list, months: pd.PeriodIndex) -> list:
    """MIN_SEGMENT_MONTHS 미만인 구간을 다음 구간과 병합 (마지막 구간이면 이전 구간에 흡수)"""
    boundaries = list(boundaries)
    changed = True
    while changed and len(boundaries) > 1:
        changed = False
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else months[-1]
            seg_len = months.get_loc(end) - months.get_loc(start) + 1
            if seg_len < MIN_SEGMENT_MONTHS:
                if i == len(boundaries) - 1:
                    boundaries.pop(i)          # 마지막 구간 -> 이전 구간에 흡수
                else:
                    boundaries.pop(i + 1)      # 다음 구간과의 경계를 지워서 병합
                changed = True
                break
    return boundaries


def build_segments(boundaries: list, months: pd.PeriodIndex) -> pd.DataFrame:
    rows = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else months[-1]
        rows.append({"segment": i + 1, "start": start, "end": end,
                      "n_months": (months.get_loc(end) - months.get_loc(start) + 1)})
    return pd.DataFrame(rows)


def main():
    monthly = load_monthly_price()
    print(f"[가격 시계열] {monthly.index.min()} ~ {monthly.index.max()}, {len(monthly)}개월")

    raw_boundaries = detect_turning_points(monthly)
    print(f"\n[1차 탐지] 확정 전환점 {len(raw_boundaries)}개 (병합 전): "
          f"{[str(b) for b in raw_boundaries]}")

    merged_boundaries = merge_short_segments(raw_boundaries, monthly.index)
    segments = build_segments(merged_boundaries, monthly.index)

    print(f"\n[최종 구간] {MIN_SEGMENT_MONTHS}개월 미만 병합 후 {len(segments)}개 구간")
    print(segments.to_string(index=False))

    segments.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
