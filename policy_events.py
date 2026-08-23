"""
정책규제 이벤트 경과일 피처 — 원래 train_model.py(트랙 B, 폐기됨)에 있던 로직을
liquidity_forward_prediction.py가 계속 쓰기 때문에 별도 유틸로 분리.

build_master_data.POLICY_EVENTS(정책 발표일 목록)를 기준으로, 특정 시점(as_of_date) 기준
"마지막 규제 발표 후 며칠 지났는지"와 "최근 1년 내 규제 발표 몇 번 있었는지"를 계산한다.
"""

import pandas as pd

from build_master_data import POLICY_EVENTS

DAYS_SINCE_NO_EVENT_SENTINEL = 9999  # as_of 시점 이전에 정책 이벤트가 한 번도 없었을 때
N_TIGHTENING_LOOKBACK_DAYS = 365     # n_tightening_past_1y 집계 기간


def policy_shock_features(as_of_date: pd.Timestamp) -> tuple[float, int]:
    event_dates = sorted(pd.to_datetime(list(POLICY_EVENTS.values())))
    past_events = [d for d in event_dates if d <= as_of_date]

    if not past_events:
        days_since = DAYS_SINCE_NO_EVENT_SENTINEL
    else:
        days_since = (as_of_date - past_events[-1]).days

    window_start = as_of_date - pd.Timedelta(days=N_TIGHTENING_LOOKBACK_DAYS)
    n_recent = sum(window_start < d <= as_of_date for d in event_dates)
    return days_since, n_recent
