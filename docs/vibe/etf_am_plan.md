# ETF Factor Exposure Analysis & EMP Factor Risk Parity Allocation

## 1. 프로젝트 개요

### 1.1 목적

한국 ETF 시장에서 각 ETF의 **time-varying factor exposure**를 추정하고,
이를 바탕으로 **Factor Risk Parity 기반 EMP(ETF Managed Portfolio) 전략**을 구성한다.

PortfolioVisualizer 등 기존 툴은 full-sample static OLS로 단일 beta를 추정하는 데 그치지만,
본 프로젝트는 **rolling regression으로 동적 beta를 추정**하여 ETF의 스타일 변화를 포착하고,
이를 연금·패시브 운용에 적합한 포트폴리오 구성 방법론으로 확장한다.

### 1.2 Research Questions

1. 한국 ETF의 JKP 팩터 노출은 시간에 따라 어떻게 변화하는가?
2. 팩터 노출이 안정적인 ETF와 불안정한 ETF는 어떻게 구별되는가?
3. ETF의 time-varying beta를 활용하면 팩터 수준의 리스크를 균등하게 배분하는 EMP를 구성할 수 있는가?
4. Factor Risk Parity EMP는 단순 동일가중 또는 시가총액가중 EMP 대비 우수한 리스크 조정 성과를 보이는가?

---

## 2. 데이터

### 2.1 ETF Return Matrix

```
R_etf : pd.DataFrame
shape : (T, N)
index : DatetimeIndex, 한국 거래일 기준 (2020-01-02 ~)
cols  : ETF ticker, e.g. ["069500", "114800", "229200", ...]
values: float, daily simple return
```

- 상장 전 구간: `NaN` 유지
- N: 수백 개 한국 ETF (주식형, 채권형, 대안 등 포함)

### 2.2 Factor Theme Return Matrix

```
R_factor : pd.DataFrame
shape    : (T, M),  M = 13
index    : DatetimeIndex, R_etf.index와 동일하게 align
cols     : 13개 JKP factor theme names
values   : float, daily factor return
```

**JKP 13 Factor Themes** (예시, 실제 사용 테마로 교체):

| # | Theme |
|---|-------|
| 1 | Accruals |
| 2 | Debt Issuance |
| 3 | Investment |
| 4 | Low Risk |
| 5 | Momentum |
| 6 | Profit Growth |
| 7 | Profitability |
| 8 | Quality |
| 9 | Seasonality |
| 10 | Short-Term Reversal |
| 11 | Size |
| 12 | Skewness |
| 13 | Value |

### 2.3 전처리 조건

- `R_etf`와 `R_factor`의 date index를 **inner join**으로 align
- ETF별 `first_valid_index()` 로 상장일 파악 → 정수 인덱스 매핑
- 상장 이후 결측치(거래 정지 등): 해당 window skip

---

## 3. 방법론

### 3.1 Rolling Time-Series Regression

날짜 t, window size w에서 ETF i의 regression:

```
y_w = R_etf[t-w+1 : t+1, i]          shape: (w,)
X_w = R_factor[t-w+1 : t+1, :]       shape: (w, M)
X_const = hstack([ones(w,1), X_w])    shape: (w, M+1)

β̂ = lstsq(X_const, y_w)              shape: (M+1,)

β̂[0]   → alpha_i(t)                  scalar, rolling intercept
β̂[1:]  → beta_i(t)                   shape: (M,), 13개 factor exposure
```

**왜 rolling time-series인가:**

- Cross-sectional regression (날짜별): factor return이 ETF 간 동일 → not identified
- Full-sample OLS: static beta → style drift 포착 불가
- Fama-MacBeth: factor risk premium λ 추정이 목적 → 본 프로젝트 목적과 다름
- **Rolling time-series**: ETF별 time-varying beta 추정 → ✅ 본 프로젝트에 적합

**Window 설정:**

| Window | 특성 |
|--------|------|
| w = 60일 (3개월) | 빠른 반응, 노이즈 많음 |
| w = 120일 (6개월) | 안정성과 반응속도의 균형 (기본) |

### 3.2 ETF별 유효 구간 처리

```python
# ETF별 첫 유효 거래일 정수 인덱스
first_valid_idx = {
    ticker: R_etf.index.get_loc(R_etf[ticker].first_valid_index())
    for ticker in R_etf.columns
}

# 각 ETF의 beta 산출 시작 t
t_first_reg(i) = first_valid_idx[i] + window - 1
```

- t < t_first_reg(i): `NaN` 유지
- t >= t_first_reg(i): beta, alpha 산출

### 3.3 구현 구조 (핵심)

```python
X_const = np.hstack([np.ones((T, 1)), R_factor.values])  # (T, M+1)

for window in [60, 120]:
    ALPHA = pd.DataFrame(np.nan, index=dates, columns=tickers)  # (T, N)
    BETAS = np.full((T, N, M), np.nan)                          # (T, N, M)
    RESID = pd.DataFrame(np.nan, index=dates, columns=tickers)  # (T, N)

    for i, ticker in enumerate(tickers):
        t0 = first_valid_idx[ticker]
        t_start = t0 + window - 1

        for t in range(t_start, T):
            X_w = X_const[t-window+1 : t+1, :]       # (w, M+1)
            y_w = R_etf.values[t-window+1 : t+1, i]  # (w,)

            if np.isnan(y_w).any():
                continue

            B, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)  # (M+1,)

            ALPHA.iloc[t, i] = B[0]
            BETAS[t, i, :]   = B[1:]
            RESID.iloc[t, i] = R_etf.values[t, i] - (B[0] + B[1:] @ R_factor.values[t, :])
```

---

## 4. 산출물 정의

| 변수 | Shape | 설명 |
|------|-------|------|
| `R_etf` | (T, N) | ETF daily return 입력 |
| `R_factor` | (T, M) | JKP factor theme return 입력, M=13 |
| `ALPHA_w60` | (T, N) | 60일 rolling alpha (intercept) |
| `ALPHA_w120` | (T, N) | 120일 rolling alpha |
| `BETAS_w60` | (T, N, M) | 60일 rolling beta |
| `BETAS_w120` | (T, N, M) | 120일 rolling beta |
| `RESID_w60` | (T, N) | 60일 rolling residual |
| `RESID_w120` | (T, N) | 120일 rolling residual |

**유효 셀 구조 예시 (ALPHA_w120):**

```
            ETF_A    ETF_B    ETF_C(신규)
2020-01-02   NaN      NaN      NaN
2020-06-30  0.0003    NaN      NaN     ← ETF_A만 120일 충족
2021-11-15  0.0001   0.0005    NaN     ← ETF_B 추가
2022-03-10  0.0002   0.0003   0.0001   ← 전체 산출
```

---

## 5. EMP Factor Risk Parity Allocation — Two-Stage Approach

### 5.1 개념: Two-Stage Framework

기존 방식(ETF 비중 w를 직접 최적화)은 N≈300 차원의 **비볼록(non-convex)** 문제로
수렴이 불안정하고 해석이 어렵다. 대신 문제를 두 단계로 분리한다.

```
Stage 1 — Model Portfolio (factor space)
  : JKP 13 factor return만으로 Factor Risk Parity 최적화
  → φ* ∈ R^13  (팩터별 목표 노출량, 순수 팩터 수준의 리스크 균등화)

Stage 2 — EMP Replication (ETF space)
  : φ*를 최대한 복제하는 ETF 비중 w 탐색 (Constrained QP)
  → w* ∈ R^N   (실현 가능한 ETF 포트폴리오)
```

이 분리의 장점:
- Stage 1은 M=13 차원의 **볼록(convex)** 문제 → 유일해, 빠른 수렴
- Stage 2는 선형 제약을 가진 **이차계획법(QP)** → cvxpy / scipy SLSQP로 해결
- φ*는 ETF 선택과 무관하게 독립적으로 해석 가능 (순수 팩터 레벨 목표)
- Stage 2에서 유동성·비중 제한 등 실무 제약을 자연스럽게 추가 가능

---

### 5.2 Stage 1 — Factor Risk Parity Model Portfolio

팩터 수익률 공분산: `Σ_f = Cov(R_factor)`, shape: (M, M)

팩터 포트폴리오 φ ∈ R^M 의 분산 및 리스크 기여:

```
σ²_φ = φ.T @ Σ_f @ φ

팩터 k의 리스크 기여:
  RC_k = φ[k] * (Σ_f @ φ)[k]
  → Σ_k RC_k = σ²_φ

목적함수 (log-barrier, 볼록):
  min_{φ} φ.T @ Σ_f @ φ  -  (1/M) * Σ_k ln(φ[k])

제약조건:
  φ[k] > 0    (long only in factor space)
  Σ_k φ[k] = 1
```

log-barrier 형식은 RC_k = σ²_φ / M 조건의 볼록 완화(convex relaxation)이며,
유일해가 보장된다. 결과 φ*는 **Model Portfolio**: 팩터 k에 얼마나 노출할지의
순수 팩터-레벨 목표 벡터.

---

### 5.3 Stage 2 — ETF Replication (Constrained QP)

날짜 t의 beta matrix: `B_t ∈ R^(N×M)` (N개 ETF, M=13 팩터)
목표 팩터 노출: `φ*` (Stage 1 결과)

ETF 포트폴리오의 팩터 노출: `B_t.T @ w ∈ R^M`

φ*와의 추적오차(factor tracking error)를 최소화:

```
목적함수 (convex QP):
  min_{w}  (B_t.T @ w - φ*).T @ Σ_f @ (B_t.T @ w - φ*)
           → 팩터 공분산으로 가중한 tracking error

제약조건:
  Σ_i w_i = 1          (fully invested)
  w_i >= 0             (long only)
  w_i <= w_max         (최대 비중 제한, e.g. 0.10)

선택적 추가 제약:
  유효관측수 < min_obs인 ETF 제외  (beta 추정 불신뢰 ETF 필터)
```

잔차: `ε_t = B_t.T @ w* - φ*` → 복제 품질 지표 (시계열로 추적 가능)

---

### 5.4 리밸런싱 설계

| 항목 | 설정 |
|------|------|
| 리밸런싱 주기 | 월별 (매월 말 영업일) |
| Σ_f 추정 | rolling window (BETAS와 동일한 w=120일) |
| Stage 1 업데이트 | 매 리밸런싱 시 재계산 (φ*는 Σ_f 변화에 따라 월별 갱신) |
| Stage 2 업데이트 | 매 리밸런싱 시 재계산 (B_t, φ* 모두 갱신됨) |
| ETF 필터 | 리밸런싱 시점 기준 유효관측수 ≥ 120인 ETF만 포함 |
| 거래비용 | 비중 변화에 비례한 비용 차감 (optional, sensitivity 분석) |

---

## 6. 분석 및 리포트 구성

### 6.1 분석 1: Time-Varying Factor Exposure 기술통계

- ETF별 beta 시계열 시각화 (주요 ETF 5~10개)
- 팩터별 cross-sectional beta 분포 변화 (boxplot over time)
- **Style Drift 지표**: beta의 rolling 표준편차 → 노출이 불안정한 ETF 탐지

### 6.2 분석 2: Factor Attribution

```
팩터 기여 수익률:
  r_attributed(t) = BETAS(t) @ R_factor(t)    shape: (N,)

잔차 알파:
  r_residual(t)   = R_etf(t) - r_attributed(t) - ALPHA(t)
```

- 팩터 기여 vs 잔차 알파의 비중 분석
- ETF 유형별(주식/채권/대안) 팩터 기여 패턴 비교

### 6.3 분석 3: Factor Risk Parity EMP 백테스트

**비교 벤치마크:**

| 전략 | 설명 |
|------|------|
| EW-EMP | 동일가중 EMP |
| MV-EMP | 최소분산 EMP |
| **FRP-EMP** | **Factor Risk Parity EMP (본 전략)** |

**평가 지표:**

- Annualized Return / Volatility / Sharpe Ratio
- Max Drawdown / Calmar Ratio
- Factor exposure concentration (Herfindahl index)
- Turnover (거래비용 sensitivity)

### 6.4 Window 민감도 분석

- w=60 vs w=120 결과 비교
- Beta 추정 안정성과 포트폴리오 성과 trade-off 분석

---

## 7. 프로젝트 구조 (파일)

```
project/
├── data/
│   ├── raw/
│   │   ├── chkxp/                   # CHKXP ETF daily CSV
│   │   └── jkp/                     # JKP factor theme CSV
│   └── db/
│       └── chkxp/etf_daily/         # parquet cache (open_etf_daily)
├── notebooks/
│   ├── 04_ETFAM_FACTORREG.ipynb     # Rolling OLS → ALPHA, BETAS, RESID
│   └── 05_ETFAM_FRP_EMP.ipynb       # Two-stage FRP + backtest
├── src/
│   ├── factor_risk_parity.py        # Stage 1: RP on factors (φ*)
│   ├── emp_replication.py           # Stage 2: ETF tracking QP (w*)
│   └── backtest.py                  # EW / MV / FRP-EMP 성과 비교
├── output/
│   ├── ALPHA_w120.parquet
│   ├── BETAS_w120.npy               # (T, N, M) — Stage 2 입력
│   ├── RESID_w120.parquet
│   ├── phi_star.parquet             # Stage 1 결과: φ* 시계열 (T_reb, M)
│   └── weights_frp.parquet          # Stage 2 결과: w* 시계열 (T_reb, N)
└── report/
    ├── report.qmd                   # Quarto 리포트 (HTML, 인터랙티브)
    └── slides.qmd                   # Quarto 슬라이드 (Reveal.js)
```

### 7.1 산출물 정의 (추가)

| 변수 | Shape | 설명 |
|------|-------|------|
| `phi_star` | (T_reb, M) | Stage 1: 월별 팩터 목표 노출 (model portfolio) |
| `weights_frp` | (T_reb, N) | Stage 2: 월별 ETF 비중 (FRP-EMP) |
| `replication_error` | (T_reb, M) | B_t.T @ w* - φ* (복제 잔차) |

### 7.2 결과물 포맷

| 포맷 | 파일 | 용도 |
|------|------|------|
| Quarto HTML Report | `report/report.qmd` | 인터랙티브 차트 (Plotly), 상세 분석 |
| Quarto Reveal.js Slides | `report/slides.qmd` | 발표용 슬라이드 |
| GitHub Pages | — | 공개 URL (Linktree 연결) |

---

## 8. 키움 연금운용 포지션과의 연결

| 공고 키워드 | 본 프로젝트 대응 |
|------------|----------------|
| 패시브펀드 퀀트 모델 리서치 | Rolling factor model로 ETF 특성 분석 |
| EMP 투자 전략 | Factor Risk Parity EMP 구성 및 백테스트 |
| 개별주식 팩터 모델 | JKP 13 factor theme 적용 |
| 차익거래 전략 | Residual alpha를 활용한 ETF 미스프라이싱 탐지로 확장 가능 |
| 투자 성과 모니터링 | Attribution 분석으로 팩터 기여 성과 분해 |

---

*본 문서는 구현 착수 기준 문서이며, 분석 결과에 따라 방법론 및 리포트 구성이 일부 변경될 수 있음.*
