"""
국토교통부 아파트 매매 실거래자료 조회 API 크롤러
- 대상: 법정동코드 11230 (서울 동대문구)
- 기간: 2025년 1월 ~ 2025년 12월

사전 준비:
1. .env 파일에 공공데이터포털에서 발급받은 "디코딩(Decoding)" 서비스키를 넣어주세요.
     MOLIT_SERVICE_KEY=발급받은_디코딩_키
2. pip install requests pandas python-dotenv openpyxl (이미 .venv에 설치됨)
"""

import os
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
SERVICE_KEY = os.getenv("MOLIT_SERVICE_KEY")

LAWD_CD = "11230"  # 서울 동대문구
REGION_NAME = "동대문구"
START_YM = (2025, 1)
END_YM = (2025, 12)
NUM_OF_ROWS = 1000


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

    # 자주 쓰는 컬럼이 있으면 앞으로 정렬 (없는 컬럼은 무시)
    preferred = [
        "aptNm", "umdNm", "jibun", "excluUseAr", "dealYear", "dealMonth",
        "dealDay", "dealAmount", "floor", "buildYear",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[ordered]

    period_str = f"{START_YM[0]}{START_YM[1]:02d}-{END_YM[0]}{END_YM[1]:02d}"
    csv_path = f"data/{period_str} {REGION_NAME}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n총 {len(df)}건 저장 완료 -> {csv_path}")


if __name__ == "__main__":
    main()
