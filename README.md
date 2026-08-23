# 동대문구 아파트 LTV/담보가치 리스크 모델

서울 동대문구를 파일럿 지역으로 삼아, 국토교통부 실거래가 데이터를 기반으로
아파트 담보가치 리스크를 분석하고 단지별 LTV 차등 적용 근거를 마련하는 프로젝트.

## 핵심 프레이밍: 은행 리스크관리의 LGD (Loss Given Default)

담보가 경매로 넘어갔을 때 "얼마나 손해를 보는가"는 가격 변동성만이 아니라
**환금성(유동성)**에도 좌우된다. 거래가 뜸한 단지는 강제매각(경매) 시 매수자를
찾기 어려워 매각가율이 더 크게 할인된다 (2019→2024 노도강 경매 4배 증가,
매각가율 92.7~103.7%→78.2~89.5% 하락 사례로 검증). 이 프로젝트는 이 인과관계를
근거로 **유동성을 LTV 차등화의 리스크 지표**로 채택한다.

- **트랙 A (GWR)**: "지금 이 단지가 왜 위험한가" — 정적 스냅샷 기반 설명모델.
  가격 변동성(MDD/CV)을 입지특성으로 설명하려 했으나 설명력이 낮아(R²~0.4대)
  **참고자료로만 사용**, 최종 LTV 근거로는 쓰지 않는다.
- **유동성 예측 (최종 채택)**: 입지특성(경사도/세대수/교통 등) + 시변 변수(상권/인구)로
  "이 단지가 앞으로 1년간 얼마나 자주 거래될 것인가"를 예측하는 walk-forward 모델
  (`liquidity_forward_prediction.py`, R²=0.687). 이 예측값이 최종 LTV 차등매핑의
  근거가 된다.
- **트랙 B (가격예측, XGBoost)**: 폐기됨. `train_model.py`는 삭제되었고, 그중
  정책규제 이벤트 피처만 `policy_events.py`로 분리되어 유동성 예측 모델에 재사용된다.

## 디렉토리 구조

```
데이터 크롤링/    원본 데이터 수집 (국토부/카카오/부동산원 API 크롤러)
근거자료/         모델링 과정에서의 변수선택·검증·진단 스크립트 (발표 근거자료)
archive/          더 이상 안 쓰지만 참고용으로 보관하는 코드/문서
data/
  ├─ 19_25_trading_data/   원본 연도별 실거래 데이터 (크롤링 산출물)
  ├─ data_geo_coding/      지오코딩 캐시/산출물
  ├─ csv/                  작업용 CSV 전체 (마스터 데이터, 모델 입출력, 진단 결과)
  └─ img/                  발표/보고용 시각화 차트
```

## 파이프라인 순서

```
1. 데이터 크롤링/apt_trading.py      국토교통부 매매 실거래가 API -> data/19_25_trading_data/
2. 데이터 크롤링/geocode_apt.py      카카오 API로 단지 주소 지오코딩 -> data/data_geo_coding/
3. 데이터 크롤링/세대수_크롤링.py     한국부동산원 API로 단지별 세대수 -> data/csv/apt_unit_count_동대문구.csv
4. 데이터 크롤링/전월세_크롤링.py     국토교통부 전월세 실거래가 API -> data/csv/전월세_2019_2025_동대문구.csv
5. build_master_data.py             7개년 실거래 데이터 클리닝 -> data/csv/master_19_25_cleaning.csv
                                     (POLICY_EVENTS 정의도 이 파일에서 관리)
6. 근거자료/spatial_autocorrelation.py   이웃단지 집계 개수(k) 선정 근거
7. gwr_analysis.py                  트랙 A: GWR 회귀 (참고자료)
   근거자료/gwr_feature_selection.py    GWR용 정적 입지변수 선별 (반경/학군/교통)
   근거자료/radius_selection.py         경사도 반경 4종 비교 시각화
   근거자료/panel_feature_selection.py  시변 변수 → 가격변동성 설명력 검증 (기각됨, R²~0.05)
8. liquidity_feature_selection.py   현재시점 유동성 설명력 검증 (정적+시변 변수)
   근거자료/liquidity_xgboost_check.py  비선형(XGBoost) 모델과 LASSO 비교
9. liquidity_forward_prediction.py  최종: 1년 후 유동성 예측 (walk-forward, R²=0.687)
   근거자료/liquidity_forward_overfitting_check.py  과적합 점검 (train/test R² 격차)
```

## 스크립트별 설명

### `데이터 크롤링/`

- **`apt_trading.py`**: 국토교통부 실거래가 공개 API(`RTMSDataSvcAptTrade`)로 동대문구
  (`LAWD_CD=11230`) 아파트 매매 실거래 데이터를 수집한다. `.env`의 `MOLIT_SERVICE_KEY` 필요.
- **`geocode_apt.py`**: 카카오 로컬 API(주소 검색)로 단지 위경도를 구한다. 단지명은 전국에
  동명 단지가 많아 신뢰할 수 없으므로 법정동+지번 조합으로 지오코딩한다. `.env`의
  `KAKAO_REST_API_KEY` 필요.
- **`세대수_크롤링.py`**: 한국부동산원 `AptIdInfoSvc`(공동주택 단지 식별정보) API로 단지별
  세대수(`UNIT_CNT`)를 수집한다. 이 API의 단지 식별체계가 `final.csv`의 `complex_id`와
  달라 주소(동+지번)로 매칭 (336개 중 305개, 95.6% 매칭 성공). 유동성 모델에서 가장 설명력
  높은 단일 변수로 확인됨.
- **`전월세_크롤링.py`**: 국토교통부 전월세 실거래가 API(`RTMSDataSvcAptRent`)로 2019~2025년
  전월세 데이터를 수집한다 (전세가율 피처 실험에 사용, 최종 채택은 안 됨).

### `build_master_data.py`
2019~2025년 7개년 실거래 데이터를 정제해 `data/csv/master_19_25_cleaning.csv`를 만든다.
이후 모든 분석(GWR, 유동성 모델)의 공통 입력. 정책 발표일 목록(`POLICY_EVENTS`)과 시장
국면 경계(`REGIME_BOUNDARIES`)도 이 파일에서 정의하며, `POLICY_EVENTS`는 `policy_events.py`가
가져다 쓴다.

### `policy_events.py`
`build_master_data.POLICY_EVENTS`를 기준으로 특정 시점의 "마지막 규제 발표 후 경과일"과
"최근 1년 내 규제 발표 횟수"를 계산하는 유틸. 원래 폐기된 `train_model.py`(트랙 B)에 있던
로직을 `liquidity_forward_prediction.py`가 계속 쓰기 위해 분리했다.

### `gwr_analysis.py` (참고자료)
트랙 A. 지리적 가중회귀(GWR)로 "왜 지금 이 단지가 위험한가"를 설명하는 정적 스냅샷 분석.
관측 단위는 단지 1개 = 1행, 종속변수는 MDD(주력)·CV(보조) 병행 산출. 실거래 데이터 단독
베이스라인과 입지변수 추가 확장 모델을 비교했으나 설명력 개선이 크지 않아(adjR² 0.39→0.43,
AICc 기준 유의한 개선 아님) **참고자료로만 사용**, 최종 LTV 근거에는 포함하지 않는다.

### `liquidity_feature_selection.py`
현재시점 유동성(`trade_activity_ratio`=거래분기비율)을 정적 입지변수(경사도/학군/교통/세대수)
와 시변 변수(상권/인구)의 단지별 시간평균으로 설명할 수 있는지 검증한다. 세대수(`unit_cnt`)가
가장 강력한 단일 변수로 확인됨.

### `liquidity_forward_prediction.py` (최종 핵심 산출물)
1년 후 유동성(`forward_activity_ratio_1y`)을 예측하는 walk-forward 검증 모델. 관측 단위는
(단지, 분기) 패널이며, XGBoost 사용. STEP0~STEP7 실험 로그가 파일 상단 docstring에 남아있다
(채택: 트레일링 활동비율/세대수/인구 + 상권지수 + 인구증가율 + 가격모멘텀 + 정책규제 경과일;
기각: 시장전체 활동수준, 12분기 트레일링, 전세가율). 최종 walk-forward R²=0.687.

### `근거자료/` (변수선택·검증·진단)
- **`spatial_autocorrelation.py`**: 이웃단지 집계 개수(k=5) 선정 근거.
- **`gwr_feature_selection.py`**: GWR에 추가할 정적 입지변수(경사도 반경/학군/교통) 스크리닝.
- **`radius_selection.py`**: 경사도 반경 4종(100/150/200/300m)의 예측력 비교 시각화.
- **`panel_feature_selection.py`**: 시변 변수 → 가격변동성 설명력 검증 (R²~0.05로 기각,
  "입지특성은 변동성보다 유동성에 더 적합하다"는 가설 전환의 근거).
- **`liquidity_xgboost_check.py`**: 유동성 예측에서 비선형(XGBoost) 모델이 LASSO 대비
  유의미하게 나은지 비교.
- **`liquidity_forward_overfitting_check.py`**: 최종 유동성 예측 모델의 train/test R² 격차를
  폴드별로 확인해 과적합 여부를 점검 (폴드가 뒤로 갈수록 격차 축소 → 과적합 아님).

### `archive/`
더 이상 파이프라인에 쓰이지 않지만 참고용으로 보관하는 코드/문서
(`visualize_market_cycle.py`, `데이터_명세서.pdf`, `클로드코드_인계_최종.md`).

## 데이터 (`data/`)

주요 산출물만 정리한다. 그 외 CSV는 각 스크립트 실행 시 재생성 가능.

| 파일 | 생성 스크립트 | 내용 |
|---|---|---|
| `csv/master_19_25_cleaning.csv` | `build_master_data.py` | 정제된 7개년 실거래 데이터 (모든 분석의 기준) |
| `csv/final.csv` | (외부 결합) | 실거래 + 입지변수(경사도/학군/교통) 결합 최종 테이블 |
| `csv/apt_unit_count_동대문구.csv` | `데이터 크롤링/세대수_크롤링.py` | 단지별 세대수 |
| `csv/전월세_2019_2025_동대문구.csv` | `데이터 크롤링/전월세_크롤링.py` | 전월세 실거래 (미채택 피처 실험용) |
| `csv/gwr_*.csv` | `gwr_analysis.py`, `근거자료/gwr_feature_selection.py` | GWR 입력·결과 |
| `csv/liquidity_*.csv` | `liquidity_feature_selection.py`, `liquidity_forward_prediction.py` | 유동성 모델 입력·결과 |
| `csv/panel_*.csv` | `근거자료/panel_feature_selection.py` | 시변변수→변동성 검증 (기각된 가설, 판단 필요 자료로 보류) |
| `csv/spatial_autocorrelation*.csv` | `근거자료/spatial_autocorrelation.py` | 이웃 k값 선정 근거 |
| `img/radius_selection.png` | `근거자료/radius_selection.py` | 경사도 반경 비교 차트 |
| `img/liquidity_forward_overfitting_check.png` | `근거자료/liquidity_forward_overfitting_check.py` | 과적합 점검 차트 |

## 환경 설정

```bash
pip install -r requirements.txt
```

`.env.example`을 참고해 `.env`에 `MOLIT_SERVICE_KEY`, `KAKAO_REST_API_KEY`를 채운다.
한국부동산원 API(`세대수_크롤링.py`)는 `MOLIT_SERVICE_KEY`를 공용으로 사용한다
(data.go.kr 계열 서비스키는 데이터셋별 개별 승인 후 동일 키 공유).
macOS에서 XGBoost 실행 시 `libomp`가 없으면 로드에 실패하므로 `brew install libomp` 필요.
