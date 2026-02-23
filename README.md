# ETF Factor Exposure Analysis & Factor Risk Parity EMP

한국 ETF 시장을 **LP(유동성 공급자) 관점**과 **AM(자산운용사) 관점** 두 가지 시각에서 분석하는 프로젝트입니다.  
ETF의 시간가변적 팩터 노출을 추정하고, 이를 기반으로 **Factor Risk Parity EMP(ETF Managed Portfolio)** 전략을 구성합니다.

---

## 프로젝트 개요

기존 포트폴리오 시각화 도구(PortfolioVisualizer 등)는 전체 기간에 대한 **정적 OLS**로 단일 베타를 추정하는 데 그칩니다.  
본 프로젝트는 **롤링 회귀(rolling OLS)** 로 동적 베타를 추정하여 ETF의 스타일 드리프트를 포착하고, 이를 연금·패시브 운용에 적합한 **팩터 수준의 리스크 균등 배분 전략**으로 확장합니다.

### Research Questions

1. 한국 ETF의 JKP 팩터 노출은 시간에 따라 어떻게 변화하는가?
2. 팩터 노출이 안정적인 ETF와 불안정한 ETF를 어떻게 구별하는가?
3. Time-varying beta를 활용하면 팩터 수준의 리스크를 균등하게 배분하는 EMP를 구성할 수 있는가?
4. Factor Risk Parity EMP는 동일가중·최소분산 EMP 대비 우수한 리스크 조정 성과를 보이는가?

---

## 데이터

| 데이터 | 설명 | Shape |
|--------|------|-------|
| ETF 일간 수익률 (`R_etf`) | 한국 ETF 194종목, 2020–2023 | (986, 194) |
| JKP 팩터 테마 수익률 (`R_factor`) | 13개 팩터 테마 일간 수익률 | (988, 13) |
| 한국 시장 수익률 (`R_mkt`) | 한국 VW 시장 수익률 | (987, 1) |
| ETF 장중 데이터 | 1분봉 OHLCV + iNAV, KODEX 200 / TIGER 반도체 | — |

**JKP 13 팩터 테마**: Accruals, Debt Issuance, Investment, Low Leverage, Low Risk, Momentum, Profit Growth, Profitability, Quality, Seasonality, Short-Term Reversal, Size, Value

- ETF 제공사: ACE, KIWOOM, PLUS, KODEX, RISE, SOL, TIGER 등 다수
- 상장 전 구간은 `NaN` 유지; ETF별 `first_valid_index()` 기반 유효 구간 관리
- `R_etf`와 `R_factor`는 inner join으로 986개 거래일 기준 정렬

---

## 분석 1 — LP 관점: 장중 ETF 괴리율·추적오차 분석 (`notebooks/02_EDA_ETFLP_data.ipynb`)

LP(유동성 공급자)의 핵심 관심사는 **ETF 시장가격이 iNAV(장중지표가치)에서 얼마나, 얼마나 자주 이탈하는가**입니다.  
장중 1분봉 데이터를 통해 이를 정밀하게 분석합니다.

### 분석 대상 ETF

| ETF | 데이터 행수 |
|-----|-----------|
| KODEX 200 | 23,730행 |
| TIGER 반도체TOP10 | 24,057행 |
| KP200 선물(미니·스탠더드) | 786,380행 |

### 주요 분석 내용

- **1분봉 OHLCV + iNAV 시계열 시각화**: ETF 종가, 매도/매수 거래량, 기초지수, iNAV 종가, 괴리율(프리미엄/디스카운트), 추적오차율을 5-패널 차트로 통합 시각화
- **단일 거래일 심층 분석**: 무작위 샘플 날짜에 대해 ETF 종가·기초지수·iNAV를 이중 y축으로 오버레이하고, 매도/매수 거래량, 괴리율·추적오차율을 정밀 분석
- **전처리 파이프라인**: 괴리율·추적오차율 스케일 정규화, 장중 수익률·iNAV 수익률·기초지수 변화율 파생 변수 생성

이 분석은 LP가 아비트라지 기회를 탐지하고 헷지 비용을 산정하는 데 필요한 기초 데이터 인프라를 구축하며, 추후 **실시간 괴리율 모니터링 시스템** 및 **ETF 미스프라이싱 탐지** 방향으로 확장 가능합니다.

---

## 분석 2 — AM 관점: Time-Varying Factor Exposure 추정 (`notebooks/04_ETFAM_FACTORREG.ipynb`)

### 방법론: Rolling Time-Series OLS

날짜 $t$, 윈도우 크기 $w$에서 ETF $i$의 회귀 모형:

$$r_{i,t} = \alpha_i(t) + \boldsymbol{\beta}_i(t)^\top \mathbf{f}_t + \varepsilon_{i,t}$$

- **피설명변수**: ETF $i$의 일간 단순수익률
- **설명변수**: 한국 VW 시장수익률 + JKP 13개 팩터 테마 수익률 (총 $M=14$)
- **추정량**: OLS (최소제곱법)

**왜 rolling time-series인가:**

| 방법 | 문제점 |
|------|--------|
| 횡단면 회귀(날짜별) | ETF 간 팩터수익률이 동일 → 식별 불가 |
| 전기간 OLS | 정적 베타 → 스타일 드리프트 포착 불가 |
| Fama-MacBeth | 팩터 리스크 프리미엄 λ 추정이 목적 → 본 프로젝트와 목적 상이 |
| **Rolling OLS** | **ETF별 time-varying beta 추정 → 적합** |

### 산출물

| 변수 | Shape | 설명 |
|------|-------|------|
| `ALPHA_w60` / `ALPHA_w120` | (986, 194) | 롤링 알파 (절편) |
| `BETAS_w60` / `BETAS_w120` | (986, 194, 14) | 롤링 팩터 로딩 |
| `RESID_w60` / `RESID_w120` | (986, 194) | 롤링 잔차 (고유 수익률) |

**유효 셀 비율**: w=60 기준 68.3%, w=120 기준 62.9% (상장 전 구간 NaN 제외)  
**계산 성능**: 윈도우당 약 5초 내 완료 (Pure NumPy 구현)

### 분석 시각화

- **JKP 팩터 간 상관관계 히트맵**: 팩터 테마 간 다중공선성 확인
- **주요 ETF 롤링 알파 시계열**: 팩터 설명 후 잔존하는 초과수익의 동태
- **팩터 노출 시계열**: ETF별 스타일 드리프트 및 팩터 노출의 시변성 시각화

---

## 분석 3 — AM 관점: Factor Risk Parity EMP (`notebooks/05_ETFAM_FRP_EMP.ipynb`)

### Two-Stage Framework 개요

N≈300 차원의 ETF 비중을 직접 최적화하면 **비볼록(non-convex)** 문제가 되어 수렴이 불안정합니다.  
대신 문제를 **팩터 공간**과 **ETF 공간** 두 단계로 분리합니다.

- **Stage 1 — Model Portfolio (factor space)**: JKP 13개 팩터 수익률만으로 Factor Risk Parity를 최적화하여 팩터별 목표 노출량 $\phi^* \in \mathbb{R}^{13}$을 산출
- **Stage 2 — EMP Replication (ETF space)**: $\phi^*$를 최대한 복제하는 ETF 비중 $w^* \in \mathbb{R}^N$을 Constrained QP로 탐색

**분리의 장점**: Stage 1은 M=13 차원의 **볼록(convex)** 문제로 유일해 보장, Stage 2는 유동성·비중 제한 등 실무 제약을 자연스럽게 추가 가능

---

### Stage 1: Factor Risk Parity (Log-Barrier 최적화)

팩터 포트폴리오 $\phi \in \mathbb{R}^M$의 팩터 $k$ 리스크 기여:

$$RC_k = \phi_k \cdot (\Sigma_f \phi)_k, \quad \sum_k RC_k = \phi^\top \Sigma_f \phi$$

**목적함수** (log-barrier, 볼록 완화):

$$\phi^* = \arg\min_{\phi} \; \phi^\top \Sigma_f \phi - \frac{1}{M} \sum_{k=1}^M \ln \phi_k \quad \text{s.t.} \quad \mathbf{1}^\top \phi = 1, \; \phi > 0$$

- **Solver**: SLSQP (Sequential Least-Squares Programming)
- **출력**: `phi_star` (42, 13) — 월별 팩터 목표 노출 시계열

---

### Stage 2: ETF Replication (Constrained QP)

날짜 $t$의 베타 행렬 $B_t \in \mathbb{R}^{N \times M}$을 이용, 팩터 추적오차 최소화:

$$w^* = \arg\min_{w} \; (B_t^\top w - \phi^*)^\top \Sigma_f (B_t^\top w - \phi^*)$$

$$\text{s.t.} \quad \mathbf{1}^\top w = 1, \quad 0 \le w_i \le 0.10$$

- **Solver**: CLARABEL (convex QP, CVXPY 인터페이스)
- **ETF 필터**: 리밸런싱 시점 기준 유효관측수 ≥ 120인 ETF만 포함
- **출력**: `weights_frp` (42, 194) — 월별 ETF 비중 시계열

---

### 백테스트 설정 및 벤치마크

| 항목 | 설정 |
|------|------|
| 분석 기간 | 2020-01-01 ~ 2023-12-31 |
| 롤링 윈도우 | 120 거래일 |
| 리밸런싱 주기 | 월말 영업일 기준 (총 42회) |
| ETF 최대 비중 | 10% (`w_max = 0.10`) |
| 평균 편입 ETF 수 (EW) | 138종목 |

**비교 전략:**

| 전략 | 설명 |
|------|------|
| **FRP-EMP** | Factor Risk Parity (본 전략) |
| EW-EMP | 동일가중 EMP |
| MV-EMP | 최소분산 EMP |

---

### 백테스트 결과

| 전략 | 연환산 수익률 | 연환산 변동성 | Sharpe | MDD | Calmar | Turnover | HHI |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FRP-EMP** | **+7.77%** | **8.16%** | **0.952** | -11.58% | **0.671** | 0.753 | 0.0574 |
| EW-EMP | +8.44% | 14.96% | 0.564 | -29.27% | 0.288 | 0.013 | 0.0073 |
| MV-EMP | +0.23% | 0.90% | 0.259 | -1.08% | 0.217 | 0.409 | 0.0580 |

**주요 시사점:**

- **FRP-EMP**는 세 전략 중 가장 높은 Sharpe Ratio(0.952)를 달성하며, MDD를 -11.58%로 억제
- EW-EMP는 절대 수익률(+8.44%)이 가장 높으나 변동성(14.96%)과 MDD(-29.27%)가 모두 최대
- MV-EMP는 변동성 최소화에 집중하여 수익률(+0.23%)과 Sharpe(0.259)가 모두 낮음
- **FRP-EMP는 팩터 수준의 리스크를 균등 배분함으로써 리스크 조정 성과 측면에서 명확한 우위**를 보임

**평가 지표 정의:**

- **Sharpe Ratio**: $(r_{ann} - r_f) / \sigma_{ann}$
- **MDD**: $\min_t (C_t / \max_{s \le t} C_s - 1)$
- **Calmar Ratio**: $r_{ann} / |\text{MDD}|$
- **Turnover**: $\frac{1}{T_{reb}} \sum_t \|w_t - w_{t-1}\|_1$
- **HHI (Herfindahl-Hirschman Index)**: $\mathbb{E}_t[\sum_i w_{i,t}^2]$, 비중 집중도 측정

---

## 프로젝트 구조

```
etf_lp_am/
├── data/
│   ├── raw/
│   │   ├── chkxp/          # Check Expert ETF 일간·장중 CSV
│   │   └── jkp/            # JKP 팩터 테마 CSV
│   └── db/
│       └── chkxp/          # Parquet 캐시 (open_etf_daily)
├── notebooks/
│   ├── 01_load_the_data.ipynb        # 데이터 수집 및 파싱
│   ├── 02_EDA_ETFLP_data.ipynb       # LP 관점: 장중 괴리율·추적오차 분석
│   ├── 03_EDA_ETFAM_data.ipynb       # AM 관점: ETF 일간 수익률 탐색
│   ├── 04_ETFAM_FACTORREG.ipynb      # Rolling OLS → ALPHA, BETAS, RESID
│   └── 05_ETFAM_FRP_EMP.ipynb        # Two-stage FRP-EMP 백테스트
├── src/
│   ├── factor_risk_parity.py         # Stage 1: RP on factors (φ*)
│   ├── emp_replication.py            # Stage 2: ETF tracking QP (w*)
│   └── backtest.py                   # EW / MV / FRP-EMP 성과 비교
├── output/
│   ├── ALPHA_w120.parquet
│   ├── BETAS_w120.npy                # (T, N, M)
│   ├── RESID_w120.parquet
│   ├── phi_star.parquet              # Stage 1: φ* 시계열 (T_reb, 13)
│   └── weights_frp.parquet           # Stage 2: w* 시계열 (T_reb, N)
├── docs/
│   └── vibe/
│       └── etf_am_plan.md            # 방법론 설계 문서
└── report/
    ├── report.qmd                    # Quarto 인터랙티브 리포트
    └── slides.qmd                    # Quarto Reveal.js 슬라이드
```

---

## 기술 스택

| 구분 | 사용 기술 |
|------|----------|
| 언어 | Python 3.12 |
| 패키지 관리 | uv |
| 수치 연산 | NumPy, Pandas |
| 최적화 | SciPy (SLSQP), CVXPY / CLARABEL |
| 시각화 | Matplotlib, Plotly |
| 데이터 저장 | Parquet (pyarrow) |
| 리포팅 | Quarto (HTML Report, Reveal.js) |

---

## 연구 의의 및 확장 방향

본 프로젝트는 **패시브 ETF 운용(EMP)**과 **팩터 기반 리스크 관리**를 결합한 실증적 프레임워크를 제시합니다.

- **팩터 Attribution**: 운용 수익을 팩터 기여와 잔차 알파로 분해하여 성과 귀인 분석 기반 마련
- **스타일 드리프트 모니터링**: 베타의 rolling 표준편차를 통해 팩터 노출이 불안정한 ETF를 실시간 탐지
- **ETF 미스프라이싱 탐지**: 잔차 알파를 활용한 차익거래 시그널 개발로 확장 가능
- **윈도우 민감도 분석**: w=60 vs w=120 비교를 통한 베타 안정성-반응속도 trade-off 최적화
- **실무 제약 추가**: 거래비용, 유동성 필터, 섹터 중립 제약 등을 Stage 2 QP에 자연스럽게 통합
