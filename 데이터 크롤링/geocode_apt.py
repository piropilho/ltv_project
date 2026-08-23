"""
카카오 로컬 API를 이용한 동대문구 아파트 단지 위경도 지오코딩

- aptNm(단지명)은 전국에 동명 단지가 많아 신뢰할 수 없으므로 사용하지 않는다.
- 대신 umdNm(법정동) + jibun(지번)으로 지번주소를 만들어 지오코딩한다.
- 4,064건 전체가 아니라 고유 주소(umdNm, jibun) 224건만 조회한 뒤 원본에 join한다.

사전 준비: .env 파일에 KAKAO_REST_API_KEY=발급받은_REST_API_키
"""

import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

KAKAO_KEY = os.getenv("KAKAO_REST_API_KEY")
ADDRESS_API_URL = "https://dapi.kakao.com/v2/local/search/address.json"
CITY = "서울특별시 동대문구"

# 지오 코딩 대상 데이터셋의 연도 선택
year = 24

SRC_CSV = f"data/19_25_trading_data/{year}_실거래_동대문구.csv"
CACHE_CSV = "data/data_geo_coding/geocode_cache_동대문구.csv"
OUT_CSV = f"data/data_geo_coding/{year}_geo_실거래_동대문구.csv"

REQUEST_INTERVAL_SEC = 0.1


def geocode_address(addr: str, headers: dict) -> tuple[float, float] | None:
    resp = requests.get(ADDRESS_API_URL, headers=headers, params={"query": addr}, timeout=10)
    resp.raise_for_status()
    docs = resp.json().get("documents", [])
    if not docs:
        return None
    doc = docs[0]
    return float(doc["y"]), float(doc["x"])  # (lat, lon)


def main():
    if not KAKAO_KEY:
        sys.exit("KAKAO_REST_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

    df = pd.read_csv(SRC_CSV)
    addr_df = df[["umdNm", "jibun"]].drop_duplicates().reset_index(drop=True)
    addr_df["address"] = CITY + " " + addr_df["umdNm"] + " " + addr_df["jibun"].astype(str)

    print(f"고유 주소 {len(addr_df)}건 지오코딩 시작")

    headers = {"Authorization": f"KakaoAK {KAKAO_KEY}"}
    lats, lons, matched = [], [], []

    for i, addr in enumerate(addr_df["address"], 1):
        try:
            result = geocode_address(addr, headers)
        except requests.exceptions.RequestException as e:
            print(f"  [{i}/{len(addr_df)}] 요청 실패: {addr} ({e})")
            result = None

        if result is None:
            lats.append(None)
            lons.append(None)
            matched.append(False)
            print(f"  [{i}/{len(addr_df)}] 매칭 실패: {addr}")
        else:
            lat, lon = result
            lats.append(lat)
            lons.append(lon)
            matched.append(True)

        time.sleep(REQUEST_INTERVAL_SEC)

    addr_df["lat"] = lats
    addr_df["lon"] = lons
    addr_df["geocode_matched"] = matched

    success_rate = addr_df["geocode_matched"].mean()
    print(f"\n지오코딩 성공률: {success_rate:.1%} ({addr_df['geocode_matched'].sum()}/{len(addr_df)})")

    addr_df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig")
    print(f"주소별 좌표 캐시 저장 -> {CACHE_CSV}")

    merged = df.merge(addr_df[["umdNm", "jibun", "lat", "lon"]], on=["umdNm", "jibun"], how="left")
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"좌표 결합된 전체 데이터 저장 -> {OUT_CSV} ({len(merged)}건)")


if __name__ == "__main__":
    main()
