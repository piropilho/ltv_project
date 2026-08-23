"""
공동주택 단지 식별정보(한국부동산원, data.go.kr 15106817) 크롤링 — 세대수 확보
====================================================================================
목적: liquidity_feature_selection.py에서 확인한 R2=0.16(정적+시변) 천장을 넘어서기 위한
      1순위 후보변수. 세대수(단지 규모)는 "어느 분기든 누군가 팔 확률"과 기계적으로
      연결돼 trade_activity_ratio 설명력을 크게 끌어올릴 가능성이 높음.

API: getAptInfo (공동주택 단지 식별정보 기본정보 조회)
     https://api.odcloud.kr/api/AptIdInfoSvc/v1/getAptInfo
     인증키는 .env의 MOLIT_SERVICE_KEY 재사용 (data.go.kr 계정당 공용 일반인증키라
     기존 국토교통부 실거래가 API에 쓰던 것과 동일)

매칭: 이 API의 COMPLEX_PK(한국부동산원 자체 단지고유번호)는 final.csv의 complex_id(내부
      순번)와 체계가 달라 직접 조인 불가. cond[ADRES::LIKE]=서울특별시 동대문구 로 동대문구
      전체를 긁은 뒤, 응답 주소(ADRES)에서 "동"+"지번"을 파싱해 final.csv의 dong/jibun과
      매칭한다.

출력: data/csv/apt_unit_count_동대문구.csv
"""

import os
import re
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.environ["MOLIT_SERVICE_KEY"]
BASE_URL = "https://api.odcloud.kr/api/AptIdInfoSvc/v1/getAptInfo"
ADDRESS_QUERY = "서울특별시 동대문구"
PER_PAGE = 100
OUT_CSV = "data/csv/apt_unit_count_동대문구.csv"

DONG_JIBUN_PATTERN = re.compile(r"(\S+동)\s+(\d+(?:-\d+)?)")


def fetch_all() -> pd.DataFrame:
    records = []
    page = 1
    while True:
        params = {
            "page": page,
            "perPage": PER_PAGE,
            "cond[ADRES::LIKE]": ADDRESS_QUERY,
            "serviceKey": SERVICE_KEY,
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        data = payload.get("data", [])
        match_count = payload.get("matchCount", 0)  # LIKE 필터에 맞는 건수 (totalCount는 전국 전체 건수라 부적절)
        records.extend(data)
        print(f"  page {page}: {len(data)}건 (누적 {len(records)}/{match_count})")

        if not data or len(records) >= match_count:
            break
        page += 1
        time.sleep(0.2)

    return pd.DataFrame(records)


def parse_dong_jibun(adres: str):
    if not isinstance(adres, str):
        return None, None
    m = DONG_JIBUN_PATTERN.search(adres)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main():
    print(f"[요청] {BASE_URL}  cond[ADRES::LIKE]={ADDRESS_QUERY}")
    df = fetch_all()
    print(f"\n[수집완료] 총 {len(df)}건")

    df = df[df["COMPLEX_GB_CD"].astype(str) == "1"].copy()
    print(f"[아파트만 필터] {len(df)}건")

    parsed = df["ADRES"].apply(parse_dong_jibun)
    df["dong"] = parsed.apply(lambda t: t[0])
    df["jibun"] = parsed.apply(lambda t: t[1])
    unmatched = df["dong"].isna().sum()
    if unmatched:
        print(f"  [경고] 주소 파싱 실패 {unmatched}건 (dong/jibun=NaN)")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[저장] -> {OUT_CSV}")


if __name__ == "__main__":
    main()
