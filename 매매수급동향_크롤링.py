"""
한국부동산원 아파트 매매수급동향지수 크롤링 (R-ONE Open API)
=========================================================
XGBoost 트랙(train_model.py)에 선행지표 두 번째 후보로 붙이기 위한 데이터 수집 스크립트.
기준금리 피처는 이미 시도 후 폐기(walk-forward 가중 R² 기준 베이스라인 미달) — 매매수급동향
지수는 거시지표(금리)가 아니라 시장참여자의 매수/매도 심리 자체를 재는 지수(0=공급우위,
200=수요우위, 100=균형)라서 다음 후보로 선정.

데이터 출처: 한국부동산원 R-ONE(부동산통계정보시스템) Open API
  통계표: A_2024_00076 ((월) 매매수급동향_아파트)
  항목: 100001 (지수)
  지역: 구 단위 데이터는 제공하지 않음. 동대문구가 속한 권역인
        CLS_ID=520011(동북권, 서울>강북지역>동북권)을 기본값으로 사용.
        참고용으로 서울 전체(CLS_ID=500008)도 함께 받아둔다.

사전 준비: .env 파일에 REB_API_KEY=발급받은_인증키
"""

import os
import sys
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

REB_API_KEY = os.getenv("REB_API_KEY")
BASE_URL = "https://www.reb.or.kr/r-one/openapi/SttsApiTblData.do"

STATBL_ID = "A_2024_00076"  # (월) 매매수급동향_아파트
DTACYCLE_CD = "MM"
ITM_ID = 100001              # 지수

REGIONS = {
    "dongbuk": 520011,  # 동북권 (서울>강북지역>동북권, 동대문구 포함) — 메인으로 쓸 값
    "seoul": 500008,    # 서울 전체 — 참고/비교용
}

START_WRTTIME = "201901"                          # 실거래 데이터 시작월과 동일하게 맞춤
END_WRTTIME = date.today().strftime("%Y%m")        # 오늘까지

PAGE_SIZE = 1000
OUT_CSV = "data/supply_demand_index_2019_present.csv"


def fetch_index(cls_id: int) -> pd.DataFrame:
    params = {
        "KEY": REB_API_KEY, "Type": "json",
        "STATBL_ID": STATBL_ID, "DTACYCLE_CD": DTACYCLE_CD,
        "CLS_ID": cls_id, "ITM_ID": ITM_ID,
        "START_WRTTIME": START_WRTTIME, "END_WRTTIME": END_WRTTIME,
        "pSize": PAGE_SIZE,
    }
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json()

    result = body["SttsApiTblData"][0]["head"][1]["RESULT"]
    if result["CODE"] != "INFO-000":
        sys.exit(f"[R-ONE API 오류] {result['CODE']}: {result['MESSAGE']}")

    rows = body["SttsApiTblData"][1]["row"]
    df = pd.DataFrame(rows)[["WRTTIME_IDTFR_ID", "DTA_VAL", "CLS_NM"]]
    df.columns = ["year_month", "supply_demand_index", "region"]
    df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")
    df["supply_demand_index"] = df["supply_demand_index"].astype(float)
    return df.sort_values("year_month").reset_index(drop=True)


def main():
    if not REB_API_KEY:
        sys.exit("REB_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    frames = []
    for label, cls_id in REGIONS.items():
        df = fetch_index(cls_id)
        print(f"[{label} / CLS_ID={cls_id}] {df['year_month'].min()} ~ {df['year_month'].max()}, "
              f"{len(df)}개월")
        print(df.tail(6).to_string(index=False))
        print()
        frames.append(df.assign(series=label))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[저장] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
