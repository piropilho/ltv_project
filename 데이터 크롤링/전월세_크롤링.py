"""
국토교통부 아파트 전월세 실거래자료 조회 API 크롤러
- 대상: 법정동코드 11230 (서울 동대문구)
- 기간: 2019년 1월 ~ 2025년 12월 (final.csv/master 매매자료와 동일 커버리지)
- 목적: 전세가율(전세가/매매가) 피처를 만들어 미래 유동성 예측(liquidity_forward_prediction.py)의
        2순위 후보변수로 테스트

apt_trading.py(매매 API 크롤러)와 동일 구조, 오퍼레이션만 RTMSDataSvcAptRent로 교체.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent"
SERVICE_KEY = os.getenv("MOLIT_SERVICE_KEY")

LAWD_CD = "11230"  # 서울 동대문구
REGION_NAME = "동대문구"
START_YM = (2019, 1)
END_YM = (2025, 12)
NUM_OF_ROWS = 1000
OUT_CSV = "data/csv/전월세_2019_2025_동대문구.csv"


def month_range(start_ym, end_ym):
    y, m = start_ym
    ey, em = end_ym
    while (y, m) <= (ey, em):
        yield f"{y:04d}{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def fetch_page(lawd_cd: str, deal_ymd: str, page_no: int, num_of_rows: int = NUM_OF_ROWS) -> ET.Element:
    params = {
        "serviceKey": SERVICE_KEY,
        "LAWD_CD": lawd_cd,
        "DEAL_YMD": deal_ymd,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise RuntimeError(f"XML 파싱 실패 ({deal_ymd}, page {page_no}): {resp.text[:300]}") from e

    result_code = root.findtext(".//resultCode")
    result_msg = root.findtext(".//resultMsg")
    if result_code is not None and result_code != "000":
        raise RuntimeError(f"API 오류 [{result_code}] {result_msg} (DEAL_YMD={deal_ymd})")

    return root


def fetch_month(lawd_cd: str, deal_ymd: str) -> list[dict]:
    items: list[dict] = []
    page_no = 1

    while True:
        root = fetch_page(lawd_cd, deal_ymd, page_no)
        body_items = root.findall(".//items/item")

        for item in body_items:
            row = {child.tag: (child.text or "").strip() for child in item}
            items.append(row)

        total_count = int(root.findtext(".//totalCount") or 0)
        if page_no * NUM_OF_ROWS >= total_count:
            break
        page_no += 1
        time.sleep(0.2)

    return items


def main():
    if not SERVICE_KEY:
        sys.exit(
            "MOLIT_SERVICE_KEY가 설정되지 않았습니다. .env 파일에 "
            "MOLIT_SERVICE_KEY=발급받은_디코딩_키 형식으로 넣어주세요."
        )

    all_rows: list[dict] = []
    for deal_ymd in month_range(START_YM, END_YM):
        print(f"[수집중] LAWD_CD={LAWD_CD} DEAL_YMD={deal_ymd}")
        try:
            rows = fetch_month(LAWD_CD, deal_ymd)
        except RuntimeError as e:
            print(f"  -> 실패: {e}", file=sys.stderr)
            continue
        print(f"  -> {len(rows)}건 수집")
        all_rows.extend(rows)
        time.sleep(0.3)

    if not all_rows:
        print("수집된 데이터가 없습니다.")
        return

    df = pd.DataFrame(all_rows)

    preferred = [
        "aptNm", "umdNm", "jibun", "excluUseAr", "dealYear", "dealMonth",
        "dealDay", "deposit", "monthlyRent", "floor", "buildYear",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n총 {len(df)}건 저장 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
