"""
한국은행 기준금리 크롤링 (ECOS Open API)
=======================================
XGBoost 트랙(train_model.py)에 선행지표로 붙이기 위한 금리 데이터 수집 스크립트.
현재는 실거래 데이터 단독으로 학습 중이며, 이 스크립트는 별도로 데이터만 받아두는
단계 — train_model.py에 실제로 결합하는 건 다음 단계에서 진행.

데이터 출처: 한국은행 경제통계시스템(ECOS) Open API
  통계표: 722Y001 (한국은행 기준금리 및 여수신금리)
  통계항목: 0101000 (한국은행 기준금리)

사전 준비: .env 파일에 ECOS_API_KEY=발급받은_API_키
"""

import os
import sys
from datetime import date

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ECOS_API_KEY = os.getenv("ECOS_API_KEY")
BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

STAT_CODE = "722Y001"   # 한국은행 기준금리 및 여수신금리
ITEM_CODE1 = "0101000"  # 한국은행 기준금리
CYCLE = "M"              # D=일별, M=월별, A=연별 — master_19_25_cleaning.csv가 월 단위라 M으로 맞춤

START_DATE = "201901"                              # 실거래 데이터 시작월(2019-01)과 동일하게 맞춤
END_DATE = date.today().strftime("%Y%m")            # 오늘까지

REQUEST_ROWS = "1/1000"  # 조회 시작/끝 행 번호 (월별 기준 1000개면 83년치라 충분)
OUT_CSV = "data/base_rate_2019_present.csv"


def fetch_base_rate() -> pd.DataFrame:
    url = (
        f"{BASE_URL}/{ECOS_API_KEY}/json/kr/{REQUEST_ROWS}/"
        f"{STAT_CODE}/{CYCLE}/{START_DATE}/{END_DATE}/{ITEM_CODE1}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    body = resp.json()

    if "RESULT" in body:
        code = body["RESULT"].get("CODE")
        msg = body["RESULT"].get("MESSAGE")
        sys.exit(f"[ECOS API 오류] {code}: {msg}")

    rows = body.get("StatisticSearch", {}).get("row")
    if not rows:
        sys.exit("[ECOS API] 응답에 데이터가 없습니다. STAT_CODE/ITEM_CODE1/기간을 확인해주세요.")

    df = pd.DataFrame(rows)[["TIME", "DATA_VALUE", "UNIT_NAME"]]
    df.columns = ["year_month", "base_rate", "unit"]
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y%m").dt.to_period("M")
    df["base_rate"] = df["base_rate"].astype(float)
    df = df.sort_values("year_month").reset_index(drop=True)
    return df


def main():
    if not ECOS_API_KEY:
        sys.exit("ECOS_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    df = fetch_base_rate()
    print(f"[조회 결과] {df['year_month'].min()} ~ {df['year_month'].max()}, {len(df)}개월")
    print(df.tail(10).to_string(index=False))

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[저장] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
