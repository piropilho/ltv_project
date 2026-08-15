# 동대문구 아파트 LTV/담보가치 리스크 모델

서울 동대문구를 파일럿 지역으로 삼아, 국토교통부 실거래가 데이터를 기반으로
아파트 담보가치 리스크를 분석하는 프로젝트. 아래 두 트랙으로 구성된다.

- **트랙 A (GWR)**: "지금 이 단지가 왜 위험한가" — 정적 스냅샷 기반 설명모델
- **트랙 B (XGBoost)**: "다음에 무슨 일이 일어날까" — 시간인지형 예측모델

두 트랙의 세부 설계 원칙은 [`XGBoost_기획안_v2.md`](XGBoost_기획안_v2.md) 참고.

## 파이프라인 순서

```
1. apt_trading.py         국토교통부 API 크롤링 -> data/*.csv
2. geocode_apt.py         카카오 API로 단지 주소 지오코딩 -> lat/lon 결합
3. apt_eda.ipynb          단일 연도(2025) 실거래 데이터 EDA
4. eda_7years.py          7개년(2019~2025) 통합 EDA + 국면(시장사이클) 구분
                           -> data/master_19_25_cleaning.csv (이후 모든 분석의 기준 데이터)
5. visualize_market_cycle.py   국면/정책이벤트를 얹은 가격·거래량 추이 시각화
6. spatial_autocorrelation.py  이웃단지 집계 개수(k) 선정 근거 진단
7. gwr_analysis.py         트랙 A: GWR 회귀 (실거래 데이터 단독)
8. train_model.py          트랙 B: XGBoost walk-forward 예측 (실거래 데이터 단독)
```

## 스크립트별 설명

### `apt_trading.py`
국토교통부 실거래가 공개 API(`RTMSDataSvcAptTrade`)를 호출해 동대문구(`LAWD_CD=11230`)
아파트 매매 실거래 데이터를 수집한다. `.env`의 `MOLIT_SERVICE_KEY`를 사용하며,
결과를 `data/{수집기간} {지역}.csv` 형식으로 저장한다.

### `geocode_apt.py`
카카오 로컬 API(주소 검색)로 아파트 단지의 위경도를 구한다. 단지명(`aptNm`)은 전국에
동명 단지가 많아 신뢰할 수 없으므로, 법정동(`umdNm`) + 지번(`jibun`) 조합으로 지오코딩하고
원본 거래 데이터에 결합한다. `.env`의 `KAKAO_REST_API_KEY` 필요.

### `apt_eda.ipynb`
2025년 단일 연도 실거래 데이터를 대상으로 한 탐색적 데이터 분석 노트북. 해제거래·중복거래
(공공기관 대량매입 등) 처리, 로그변환 여부를 실제 왜도(skewness) 수치로 판단하는 등
"근거 없이 관행적으로 변환하지 않는다"는 원칙을 따른다.

### `eda_7years.py`
2019~2025년 7개년 통합 데이터를 정제하고, 시장 국면(`REGIME_BOUNDARIES`)과 정책 발표일
(`POLICY_EVENTS`)을 정의한다. 국면 경계 중 정책 발표에 맞춰야 하는 것(2020, 2025 규제)은
고정값으로, 유기적 전환점(상승기→급락기, 급락기→회복기)은 실제 월별 평균가의 고점(2021-09)/
저점(2022-10)을 근거로 데이터 기반으로 잡았다. MDD(최대낙폭)와 CV(변동계수)를 국면별로
함께 산출해 서로 다른 리스크 측면을 비교한다. 이 스크립트의 산출물
`data/master_19_25_cleaning.csv`가 이후 모든 분석(GWR, XGBoost)의 공통 입력이다.

### `visualize_market_cycle.py`
`eda_7years.py`의 `REGIME_BOUNDARIES`/`POLICY_EVENTS`를 가져와 월별 평균가·거래량 추이 위에
국면 음영과 정책 발표일 점선을 얹은 차트(`data/market_cycle.png`)를 그린다.

### `spatial_autocorrelation.py`
이웃단지 집계 시 몇 개(k)를 묶을지 통계적으로 정하기 위한 진단 스크립트. 단지 쌍 간 거리와
가격 상관관계로 경험적 상관로그램(+준분산)을 구하고, k=3/5/8/10/15/20 각각이 평균적으로
어느 반경까지 이웃을 끌어오는지, 그 반경에서 상관관계가 얼마인지 표로 정리한다.
`N_NEIGHBORS=5`(GWR·XGBoost 공통 채택값)가 상관관계가 충분히 높은 구간(반경 약 0.2km,
r=0.41) 안에 안전하게 들어간다는 근거를 이 스크립트로 확인했다.

### `gwr_analysis.py`
트랙 A. 지리적 가중회귀(GWR)로 "왜 지금 이 단지가 위험한가"를 설명하는 정적 스냅샷 분석.
관측 단위는 단지 1개 = 1행이며, 종속변수는 MDD(주력)·CV(보조)를 병행 산출한다. 검증셋 분리
없이 국지적 유의성(t-value)으로 판단하며, 학군/지하철 등 입지변수는 아직 결합하지 않고
실거래 데이터만으로 베이스라인을 산출한다. 이웃단지 가격수준(`neighbor_price_level`)·
모멘텀(`neighbor_momentum_3m`) 피처는 k=5 최근접 이웃(BallTree, haversine 거리) 기준.

### `train_model.py`
트랙 B. XGBoost로 "다음에 무슨 일이 일어날지"를 예측하는 시간인지형 모델. 관측 단위는
(단지, 시점) 패널이며, 개별 단지의 월별 거래가 희박해 가격 관련 피처/타겟은 전부
"단지 + k=5 최근접 이웃"의 트레일링 평균으로 스무딩한다. 타겟은 t 시점 대비
`HORIZON_MONTHS`(기본 6개월) 뒤 지역 스무딩 가격의 변화율(%). 검증은 walk-forward
(확장윈도우)만 사용하고 random split은 금지하며, 가장 최근 구간은 폴드 구성에서 완전히
제외한 뒤 최종 1회 평가에만 쓴다. 정책 충격 변수는 개별 이벤트 더미 대신
`days_since_last_tightening`/`n_tightening_past_1y`처럼 일반화된 형태로 정의해
`eda_7years.py`의 `POLICY_EVENTS`를 재사용한다. 모든 튜닝 가능한 값(윈도우 길이, k, 호라이즌,
폴드 구성 등)은 파일 상단에 상수로 노출되어 있다.

## 데이터 (`data/`)

주요 산출물만 정리한다. 그 외 원본/중간 CSV는 각 스크립트 실행 시 재생성 가능.

| 파일 | 생성 스크립트 | 내용 |
|---|---|---|
| `master_19_25_cleaning.csv` | `eda_7years.py` | 정제된 7개년 실거래 데이터 (모든 분석의 기준) |
| `market_cycle.png` | `visualize_market_cycle.py` | 국면 구분 가격·거래량 추이 차트 |
| `spatial_autocorrelation*.csv` | `spatial_autocorrelation.py` | 이웃 k값 선정 근거 (상관로그램, k별 반경표) |
| `gwr_complex_table.csv` | `gwr_analysis.py` | GWR 입력용 단지별 정적 스냅샷 테이블 |
| `xgb_panel.csv` | `train_model.py` | XGBoost 학습용 (단지, 시점) 패널 |
| `xgb_walkforward_results.csv` | `train_model.py` | walk-forward 폴드별 성능(MAE, R²) |

## 환경 설정

```bash
pip install -r requirements.txt
```

`.env.example`을 참고해 `.env`에 `MOLIT_SERVICE_KEY`, `KAKAO_REST_API_KEY`를 채운다.
macOS에서 XGBoost 실행 시 `libomp`가 없으면 로드에 실패하므로 `brew install libomp` 필요.
