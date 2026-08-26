"""
데모: 아파트명을 입력하면 구매 시점 기준 단지 정보와 제안 LTV를 보여준다.
==========================================================================
ltv_scoring.py가 미리 계산해둔 결과(data/csv/ltv_assignment.csv)를 조회만 하는
가벼운 CLI 데모 -- 실행할 때마다 모델을 재학습하지 않는다. 예측값을 최신 데이터로
갱신하려면 ltv_scoring.py를 먼저 재실행해서 data/csv/ltv_assignment.csv를 새로 만든다.

기준 시점은 ltv_scoring.py가 예측에 사용한 마지막 관측 분기(2025-12)로 고정되어
있다 -- "오늘 이 단지를 산다면"이 아니라 "데이터가 확보된 가장 최근 시점 기준"이라는
점을 데모 출력에도 명시한다 (정책이벤트 경과일 등 시간 피처의 외삽 문제를 피하기 위한
설계, liquidity_forward_prediction.py STEP2 참고)
"""

import pandas as pd

LTV_ASSIGNMENT_CSV = "data/csv/ltv_assignment.csv"
FINAL_CSV = "data/csv/final.csv"
LATEST_YYYYMM = 202512


def load_table() -> pd.DataFrame:
    final = pd.read_csv(FINAL_CSV).sort_values(["complex_id", "yyyymm"])
    final["avg_price_ffill"] = final.groupby("complex_id")["avg_price_per_m2"].ffill()

    latest = final[final["yyyymm"] == LATEST_YYYYMM][[
        "complex_id", "apt_name", "dong", "jibun", "avg_price_ffill", "building_age",
        "station_name", "station_zone", "walk_time_min", "school_pc1", "slope_mean_300m",
    ]]

    ltv = pd.read_csv(LTV_ASSIGNMENT_CSV).drop(columns=["apt_name"])
    table = latest.merge(ltv, on="complex_id", how="left")
    return table


def search(table: pd.DataFrame, query: str) -> pd.DataFrame:
    return table[table["apt_name"].str.contains(query, case=False, na=False)].reset_index(drop=True)


def parse_price(text: str) -> float | None:
    text = text.strip().replace(",", "").replace("만원", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        print("  (숫자를 인식하지 못해 매매가 없이 진행합니다)")
        return None


def show_profile(row: pd.Series, purchase_price: float | None):
    print("=" * 62)
    print(f" {row['apt_name']}  ({row['dong']} {row['jibun']})")
    print("=" * 62)
    print(f" 기준 시점         : {LATEST_YYYYMM // 100}년 {LATEST_YYYYMM % 100}월 (데이터 확보 최근 분기)")

    unit_cnt = row["unit_cnt"]
    print(f" 세대수            : {unit_cnt:,.0f}세대" if pd.notna(unit_cnt) else " 세대수            : 정보 없음")
    print(f" 준공 후 경과연수  : {row['building_age']:.0f}년")
    print(f" 최근 관측 시세    : {row['avg_price_ffill']:,.0f}만원/㎡")
    print(f" 역세권            : {row['station_zone']} ({row['station_name']}, 도보 {row['walk_time_min']:.0f}분)")
    print("-" * 62)

    if pd.isna(row["assigned_ltv_pct"]):
        print(" 예측 유동성/LTV   : 세대수 데이터 미매칭으로 산출 불가")
        print("=" * 62)
        return

    print(f" 예측 유동성(향후1년): {row['predicted_liquidity_1y']:.1%}  "
          f"(동대문구 내 상위 {100 - row['liquidity_percentile']:.0f}%)")
    print(f" 제안 LTV          : {row['assigned_ltv_pct']:.1f}%  (정부 상한 40% 이내, 백분위 연속 차등)")

    if purchase_price is not None:
        loan = purchase_price * row["assigned_ltv_pct"] / 100
        print("-" * 62)
        print(f" 희망 매매가        : {purchase_price:,.0f}만원")
        print(f" 예상 대출가능금액  : {loan:,.0f}만원  (매매가 x LTV {row['assigned_ltv_pct']:.1f}%)")
    print("=" * 62)


def pick_match(matches: pd.DataFrame) -> pd.Series | None:
    if len(matches) == 1:
        return matches.iloc[0]

    print(f"\n검색결과 {len(matches)}건:")
    for i, r in matches.iterrows():
        print(f"  [{i}] {r['apt_name']}  ({r['dong']} {r['jibun']})")
    sel = input("번호를 선택하세요: ").strip()
    if not sel.isdigit() or int(sel) not in range(len(matches)):
        print("잘못된 선택입니다.\n")
        return None
    return matches.iloc[int(sel)]


def main():
    table = load_table()
    print("동대문구 아파트 LTV 제안 데모")
    print(f"(조회 가능 {len(table)}개 단지 -- 종료: q)\n")

    while True:
        try:
            query = input("아파트명을 입력하세요: ").strip()
        except EOFError:
            break
        if query.lower() == "q":
            break
        if not query:
            continue

        matches = search(table, query)
        if matches.empty:
            print(f"'{query}'로 검색된 단지가 없습니다.\n")
            continue

        row = pick_match(matches)
        if row is None:
            continue

        price_text = input("희망 매매가(만원, 생략 가능): ")
        purchase_price = parse_price(price_text)

        print()
        show_profile(row, purchase_price)
        print()


if __name__ == "__main__":
    main()
