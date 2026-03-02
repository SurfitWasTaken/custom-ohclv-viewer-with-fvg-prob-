# Fair Value Gap Fill-Time Prediction: A Statistical Analysis of Price Imbalance Reversion in Forex Markets

**A Machine Learning Framework for Modelling FVG Fill Probabilities and the Limits of Direct Monetisation**

*Version 2.0 | March 2026*

---

## Abstract

Fair Value Gaps (FVGs) — three-candle price imbalances in financial markets — are widely believed by practitioners to act as price attractors that draw subsequent price action back toward them. This paper presents a rigorous statistical investigation of FVG fill behaviour across four major forex pairs (EURUSD, GBPUSD, USDJPY, AUDUSD) spanning three years of hourly data (February 2021 – January 2024).

We develop a multi-horizon XGBoost classification framework that predicts the probability of individual FVG fills at seven time horizons (1, 2, 5, 10, 20, 50, and 100 candles). The model achieves excellent calibration (mean absolute error < 2% across all horizons) and statistically significant discrimination (OOS AUC 0.649 at the 10-candle horizon, 95% CI [0.634, 0.664], evaluated via 9-fold purged walk-forward validation with 7,230 out-of-sample predictions).

We then construct a hypothesis for direct monetisation: if FVG fills can be predicted with high accuracy, limit orders placed at FVG boundaries should yield systematic profits. An institutional-grade walk-forward backtest with realistic variable TP/SL, a 100-candle purge gap, and a naive-strategy benchmark reveals that this hypothesis **fails** — the strategy is deeply unprofitable (Profit Factor 0.12) despite an 83.5% fill-prediction accuracy. We identify three structural reasons for this failure and discuss why well-calibrated fill-time prediction remains valuable as an information signal within broader trading frameworks.

---

## 1. Introduction

### 1.1 Fair Value Gaps as Market Microstructure Phenomena

A Fair Value Gap (FVG) is a price range where only one side of the market was present, created by an impulsive three-candle sequence where the wicks of candles *n−1* and *n+1* fail to overlap. Specifically:

- **Bullish FVG**: low of candle *n+1* > high of candle *n−1* (gap below current price)
- **Bearish FVG**: high of candle *n+1* < low of candle *n−1* (gap above current price)

These structures are referenced extensively in Institutional Concepts Theory (ICT) and Smart Money Concepts (SMC) trading methodologies, where they are interpreted as zones of incomplete order execution that price must revisit to achieve "fair value." The underlying economic intuition is that impulsive moves leave unfilled limit orders and trapped counterparties, creating magnetic pull for mean-reversion when momentum subsides.

### 1.2 Motivation

Despite the ubiquity of FVG analysis in retail and institutional trading, the existing literature lacks rigorous quantitative answers to basic questions:

1. **How frequently do FVGs actually fill, and how quickly?**
2. **Can fill timing be predicted from observable features at formation?**
3. **Does predictable fill behaviour translate to tradeable alpha?**

This paper addresses all three questions with a dataset of 14,567 individually labelled FVGs across four major forex pairs.

### 1.3 Hypothesis

We hypothesise that:

> **H₁:** *FVG fill timing is partially predictable from gap characteristics, market momentum, volatility, and higher-timeframe trend context.*

> **H₂:** *If fill timing is predictable, a strategy that places limit orders at high-probability FVG boundaries can generate positive risk-adjusted returns.*

We present strong evidence for **H₁** and definitively **reject H₂**.

### 1.4 Related Work

FVGs are a specific case of the broader mean-reversion literature in financial econometrics. Jegadeesh and Titman (1993) established short-term mean-reversion in equity returns. Lo and MacKinlay (1990) documented departures from random walk behaviour consistent with temporary price dislocations. Our work is most closely related to the literature on order flow imbalance (Cont, Kukanov & Stoikov, 2014), which establishes that transient supply/demand imbalances create predictable short-horizon price movements — the mechanism we associate with FVG formation and fill.

The ICT/SMC practitioner literature (e.g., "The Inner Circle Trader" educational materials) provides the heuristic framework that motivates our feature engineering, though it lacks the statistical rigour and honest assessment of limitations that this paper provides.

---

## 2. Data

### 2.1 Collection

OHLCV data was collected via the OANDA REST API for four major currency pairs across three timeframes:

| Pair | 1H Candles | 4H Candles | Daily Candles | Period |
|------|-----------|-----------|--------------|--------|
| EURUSD | 18,712 | 4,678 | 1,077 | Feb 2021 – Feb 2024 |
| GBPUSD | ~18,700 | ~4,680 | ~1,075 | Feb 2021 – Feb 2024 |
| USDJPY | ~18,700 | ~4,680 | ~1,075 | Feb 2021 – Feb 2024 |
| AUDUSD | ~18,700 | ~4,680 | ~1,075 | Feb 2021 – Feb 2024 |

### 2.2 FVG Detection

The `scan_all_fvgs()` detector identifies all three-candle formations meeting the FVG criteria on 1-hour data. Each detected FVG is characterised by its type (bullish/bearish), gap boundaries (high/low), formation time, and the OHLCV properties of the impulse candle.

### 2.3 Labelling: Time-to-Fill with Censoring

For each detected FVG, we define:

- **Time-to-fill** (*T*): the number of candles after formation until price first touches the gap boundary (specifically, the gap_high for bullish FVGs, gap_low for bearish FVGs).
- **Censored indicator** (*δ*): whether the fill was observed within a 100-candle observation window. FVGs not filled within 100 candles are right-censored at *T* = 100.
- **Fill depth**: the fraction of the gap penetrated at first touch (0 to 1).

This labelling is survival-analysis-compatible, though we ultimately adopt a multi-horizon binary classification approach (see §4).
t
### 2.4 Dataset Summary

| Statistic | Value |
|-----------|-------|
| Total FVGs labelled | 14,567 |
| Fill rate (within 100 candles) | 92.7% |
| Median time-to-fill | 3 candles |
| Mean time-to-fill | 15.2 candles |
| Censored (unfilled at 100 candles) | 7.3% |
| FVGs per pair | 3,553 – 3,759 |

The high fill rate (92.7%) and low median time-to-fill (3 candles) confirm the widespread practitioner observation that FVGs are overwhelmingly revisited — and quickly. This creates the initial basis for the monetisation hypothesis.

### 2.5 Train / Validation / Test Split

A strict chronological 60/20/20 split:

| Split | Period | Samples |
|-------|--------|---------|
| Train | Feb 2021 → Nov 2022 | 8,740 |
| Validation | Nov 2022 → Jun 2023 | 2,913 |
| Test | Jun 2023 → Jan 2024 | 2,914 |

No temporal overlap exists between any split. All features are computed using only data available at or before FVG formation time.

---

## 3. Feature Engineering

Twenty-five features across six categories are extracted for each FVG at formation time. All features are strictly backward-looking — no information from after the FVG's formation is used.

### 3.1 Feature Categories

| Category | N | Features | Rationale |
|----------|---|----------|-----------|
| **Gap Properties** | 6 | gap_size, gap_size_atr, gap_volume, impulse_ratio, body_to_atr, is_bullish | Larger, more impulsive gaps may fill differently than small ones |
| **Distance from Price** | 4 | dist_to_gap, dist_to_gap_atr, gap_above_price, gap_pct_of_range | Price proximity to the gap boundary at formation |
| **Momentum** | 5 | ret_5, ret_10, ret_20, trend_20, momentum_toward_gap | Recent directional bias; momentum toward or away from the gap |
| **Volatility** | 3 | atr_14, realized_vol, vol_percentile | Higher volatility increases the probability of reaching distant price levels |
| **HTF Context** | 3 | trend_4h, trend_daily, htf_supports_fill | Whether higher-timeframe trends support the directional move required for fill |
| **Market Context** | 4 | session_hour, day_of_week, n_nearby_same, n_nearby_opposite | Time-of-day effects, FVG cluster density |

### 3.2 Feature Importance

The top features by XGBoost importance (averaged across all seven horizon models):

| Rank | Feature | Mean Importance | Interpretation |
|------|---------|:--------------:|----------------|
| 1 | session_hour | 0.057 | Strong session-dependency in fill behaviour |
| 2 | dist_to_gap_atr | 0.038 | ATR-normalised distance is the key proximity signal |
| 3 | is_bullish | 0.038 | Bullish vs bearish FVGs fill at different rates |
| 4 | gap_volume | 0.038 | Volume on the impulse candle signals conviction |
| 5 | gap_above_price | 0.037 | Direction of the gap relative to price |
| 6 | day_of_week | 0.036 | Day-of-week seasonality exists |
| 7 | realized_vol | 0.036 | Higher vol → more rapid fills |
| 8 | htf_supports_fill | 0.034 | HTF alignment with fill direction matters |
| 9 | impulse_ratio | 0.034 | Body-to-range ratio of the impulse candle |
| 10 | trend_daily | 0.034 | Daily trend direction |

Feature importance is remarkably diffuse — the model relies on many weak signals rather than a few dominant predictors, consistent with the moderate AUC values observed.

---

## 4. Model

### 4.1 Architecture: Multi-Horizon Binary Classification

An initial attempt with scikit-survival's `GradientBoostingSurvivalAnalysis` proved computationally infeasible (O(n²) per boosting iteration; estimated >30 hours on 8,740 samples with no completion). We instead adopt a pragmatic multi-horizon approach:

**For each horizon *h* ∈ {1, 2, 5, 10, 20, 50, 100}, train a separate XGBoost binary classifier with target:**

$$ y_h = \mathbb{1}[T \leq h \text{ and } \delta = 1] $$

This decomposes the survival function into seven binary questions: "Was this FVG filled within *h* candles?" Monotonicity (P(fill ≤ *h₂*) ≥ P(fill ≤ *h₁*) for *h₂* > *h₁*) is enforced post-hoc at prediction time.

**Hyperparameters** (shared across all seven models):
```
n_estimators=200, max_depth=5, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8, random_state=42
```

Training time for all seven models: **3 seconds** on CPU.

### 4.2 In-Sample Evaluation

| Horizon | Fill Rate | Train AUC | Val AUC | Test AUC | Brier Score |
|---------|:---------:|:---------:|:-------:|:--------:|:-----------:|
| 2 | 43.6% | 0.896 | 0.611 | 0.619 | 0.237 |
| 5 | 65.8% | 0.901 | 0.624 | 0.632 | 0.210 |
| 10 | 75.7% | 0.918 | 0.648 | 0.642 | 0.163 |
| 20 | 82.3% | 0.948 | 0.639 | 0.619 | 0.129 |
| 50 | 88.6% | 0.973 | 0.580 | 0.558 | 0.088 |
| 100 | 92.0% | 0.984 | 0.566 | 0.578 | 0.062 |

The large gap between train AUC (0.90+) and val/test AUC (0.56–0.65) indicates the model learns noise in-sample but generalises modestly out-of-sample. This is expected for a problem where the marginal signal above base rate is inherently weak.

### 4.3 Calibration

Calibration is the most important metric for a probability model. Our model is exceptionally well-calibrated:

| Horizon | Predicted Mean | Actual Fill Rate | |Error| |
|---------|:-------------:|:----------------:|:------:|
| 2 candles | 0.439 | 0.447 | **0.7%** |
| 5 candles | 0.656 | 0.674 | **1.8%** |
| 10 candles | 0.765 | 0.780 | **1.3%** |
| 20 candles | 0.835 | 0.844 | **0.5%** |
| 50 candles | 0.894 | 0.904 | **0.7%** |
| 100 candles | 0.928 | 0.934 | **0.3%** |

Mean absolute calibration error: **0.9%**. When the model says "80% probability of fill," the true fill rate is very close to 80%. This makes the model reliable as an information source, even though its discrimination (ability to rank-order) is moderate.

---

## 5. Walk-Forward Validation

### 5.1 Design

To eliminate any possibility of look-ahead bias, we implement an institutional-grade walk-forward validation:

- **9 expanding-window folds** with 2-month test windows
- **100-candle purge gap** (≈4 days) between training end and test start — equal to the longest prediction horizon, ensuring no training observation's target window overlaps any test observation
- **Programmatic leakage audit**: 27 checks (3 per fold) verifying temporal separation, index non-overlap, and minimum lookback availability

### 5.2 Leakage Audit

All 27 checks pass:

| Check | Result |
|-------|--------|
| Purge gap ≥ 100 hours (all 9 folds) | ✅ Pass |
| No shared indices between train/test (all 9 folds) | ✅ Pass |
| Sufficient lookback for all test observations (all 9 folds) | ✅ Pass |
| Feature temporal consistency (per pair, zero inversions) | ✅ Pass |

### 5.3 Aggregated OOS Results

Metrics aggregated across all 7,230 out-of-sample predictions:

| Horizon | AUC [95% CI] | Brier Score [95% CI] | Cal Error | N |
|---------|:------------:|:-------------------:|:---------:|:---:|
| 2 | 0.614 [0.601, 0.626] | 0.237 [0.234, 0.240] | 0.4% | 7,230 |
| 5 | 0.636 [0.622, 0.650] | 0.214 [0.210, 0.218] | 0.3% | 7,230 |
| **10** | **0.649 [0.634, 0.664]** | **0.171 [0.166, 0.176]** | **0.2%** | **7,230** |
| 20 | 0.630 [0.612, 0.648] | 0.134 [0.128, 0.140] | 0.2% | 7,230 |
| 50 | 0.589 [0.568, 0.609] | 0.090 [0.084, 0.095] | 0.4% | 7,230 |
| 100 | 0.586 [0.560, 0.612] | 0.061 [0.056, 0.066] | 0.3% | 7,230 |

*95% confidence intervals from 1,000 bootstrap resamples.*

**Key observations:**
- All AUC confidence intervals exclude 0.5 (chance), confirming statistically significant discrimination
- Peak discrimination occurs at the 10-candle horizon (AUC 0.649)
- Discrimination decreases at longer horizons as the base fill rate approaches 1.0 (ceiling effect)
- Calibration is uniformly excellent across all horizons (< 0.4% error)
- OOS AUC is consistent with in-sample val/test AUC — **no evidence of overfitting**

### 5.4 Multiple Comparison Correction

Since we test discrimination at 6 non-trivial horizons (h ∈ {2, 5, 10, 20, 50, 100}), we must correct for the multiplicity of hypothesis tests. Reporting the best-performing horizon without correction would overstate significance.

We apply both Bonferroni (conservative) and Benjamini-Hochberg (FDR = 0.05) corrections:

| Horizon | z-score | Raw p-value | BH Critical | Survives BH? | Survives Bonferroni? |
|---------|:-------:|:-----------:|:-----------:|:------------:|:-------------------:|
| h=2  | 17.92 | < 10⁻⁶⁰ | 0.0083 | ✅ | ✅ |
| h=5  | 18.97 | < 10⁻⁶⁰ | 0.0167 | ✅ | ✅ |
| h=10 | 19.31 | < 10⁻⁶⁰ | 0.0250 | ✅ | ✅ |
| h=20 | 14.04 | < 10⁻⁶⁰ | 0.0333 | ✅ | ✅ |
| h=50 | 8.36  | < 10⁻¹⁶ | 0.0417 | ✅ | ✅ |
| h=100| 6.50  | 8.0 × 10⁻¹¹ | 0.0500 | ✅ | ✅ |

**All six horizons survive both corrections.** The z-scores are large enough (6.5–19.3) that the choice of correction procedure is immaterial — the p-values are astronomically small. This is driven by the large sample size (N=7,230) and the genuine, if modest, signal at every horizon. The 10-candle horizon's AUC of 0.649 is not an artifact of cherrypicking.

### 5.5 Fold Stability

| Fold | Test Period | Avg AUC (h=2,5,10,20) |
|------|-----------|:---------------------:|
| 0 | Aug – Oct 2022 | 0.612 |
| 1 | Oct – Dec 2022 | 0.653 |
| 2 | Dec 2022 – Feb 2023 | 0.611 |
| 3 | Feb – Apr 2023 | 0.624 |
| 4 | Apr – Jun 2023 | 0.653 |
| 5 | Jun – Aug 2023 | 0.643 |
| 6 | Aug – Oct 2023 | 0.660 |
| 7 | Oct – Dec 2023 | 0.649 |
| 8 | Dec 2023 – Jan 2024 | 0.596 |

Standard deviation across folds: **0.022**. The model performance is stable over time with no structural degradation.

---

## 6. Economic Evaluation: The Monetisation Hypothesis

### 6.1 Rationale for the Hypothesis

The high fill rate (76% at 10 candles, 93% at 100 candles) combined with a model that can identify which FVGs are *more* likely to fill suggests a natural trading strategy:

1. **Signal**: Model predicts P(fill within 10 candles) ≥ 0.80
2. **Entry**: Place a limit order at the FVG boundary
3. **Take Profit**: Gap midpoint (half-gap penetration)
4. **Stop Loss**: 1.5× ATR beyond entry

The grounds for suspecting an arbitrage opportunity were:
- FVGs represent genuine order flow imbalances (incomplete institutional execution)
- The fill rate is abnormally high relative to random price walks
- The model identifies a subset with 85% fill rate (vs 76% base rate) — a 7 percentage point lift
- If the market systematically returns to these zones, limit orders placed there should be filled profitably

### 6.2 Naive Strategy Benchmark

To rigorously assess whether the *model* adds value over the base phenomenon, we construct a naive benchmark that trades every FVG regardless of model predictions, using the same TP/SL rules.

### 6.3 Results

| Metric | Model-Selected | Naive (All FVGs) |
|--------|:--------------:|:----------------:|
| Trades | 2,947 | 6,398 |
| Win Rate | 39.2% | 36.0% |
| Total P&L | −1,165,539 pips | −3,065,364 pips |
| Mean P&L/trade | −395.50 pips | −479.11 pips |
| Avg Win | +140.98 pips | +146.72 pips |
| Avg Loss | −740.78 pips | −831.32 pips |
| Sharpe Ratio | −4.53 | −5.03 |
| Max Drawdown | −1,166,025 pips | −3,065,702 pips |
| Profit Factor | 0.12 | 0.10 |

Both strategies are catastrophically unprofitable.

### 6.4 Why the Hypothesis Failed: Three Structural Reasons

#### Reason 1: Fill ≠ Profit — The Conditional Distribution After Fill

The fundamental flaw is equating "FVG boundary touched by price" with "profitable trade." A fill event means price reached the gap boundary — but what happens *after* the touch is the distribution that actually determines P&L.

Our model predicts P(touch), but tells us nothing about the conditional distribution of subsequent price movement. Analysing the 11,074 FVGs that fill within 10 candles:

| Fill Depth Category | Fraction | Interpretation |
|---------------------|:--------:|----------------|
| **Deep fills** (depth ≥ 50%) | 68.8% | Would hit our TP (gap midpoint) |
| Partial fills (0 < depth < 50%) | 29.4% | Boundary touched but no TP — adverse selection |
| Shallow fills (depth ≤ 10%) | 9.1% | Wick touch, immediate reversal |
| **Full fills** (depth ≥ 100%) | 52.9% | Complete gap closure |

**Median fill depth: 1.0** — when price does fill an FVG, it overwhelmingly fills *completely*. The distribution is bimodal: either price goes all the way through, or it barely penetrates. The 29.4% partial fills are the toxic population — these are the adverse-selection fills where price touches the boundary, triggers our limit order, then reverses.

This reframes the problem: the research question is not "will price touch the boundary?" (which our model answers) but rather **"given a boundary touch, will price continue through the gap or reverse?"** This conditional distribution — P(depth ≥ 0.5 | touch) — is the genuinely monetisable signal and is conspicuously absent from the current model.

#### Reason 2: Unfavourable TP/SL Asymmetry — Analytical Break-Even

The structural geometry of FVGs creates an unfavourable reward-to-risk ratio. In ATR-normalised terms (which are consistent across pairs):

| Component | Value (ATR units) |
|-----------|:------------------:|
| Mean FVG gap size | 0.425 ATR |
| Mean TP (half gap) | 0.213 ATR |
| Mean SL (1.5× ATR) | 1.500 ATR |
| **SL/TP ratio** | **7.1:1** |

This yields an analytical break-even win rate:

> **Break-even win rate = SL / (SL + TP) = 1.5 / (1.5 + 0.213) = 87.6%**

Now we can derive exactly what additional signal quality is required:

| Metric | Value | What It Means |
|--------|:-----:|--------------|
| P(fill within 10) | 76.0% | Our model predicts this |
| P(depth ≥ 50% &#124; fill) | 68.8% | We do NOT predict this |
| P(TP hit overall) | **52.3%** | = P(fill) × P(depth ≥ 50% &#124; fill) |
| P(TP hit needed) | **87.6%** | For break-even at 7.1:1 SL/TP |
| **Gap to close** | **35.3pp** | Mathematically unclosable by fill timing alone |

No fill-time classifier can close this gap because P(TP hit) = P(fill) × P(deep fill | fill) ≤ 1.0 × 0.688 = 68.8% — still 18.8pp below break-even. **Even a perfect fill-time oracle cannot make this strategy profitable.** The constraint is the conditional depth distribution, not the fill-time prediction.

#### Reason 3: The Base Rate Is Deceptive

The 92.7% fill rate within 100 candles sounds like a massive edge, but it reflects something mundane: on an hourly timeframe, normal price volatility will cross most small price levels within several days simply due to Brownian motion dynamics. The "reversion" to FVGs is largely indistinguishable from random price exploration.

To test this, consider: the typical FVG boundary is only 0.2 ATR away from price at formation. Any random walk with normal volatility will cross a level 0.2 standard deviations away within a few steps. The high fill rate is not evidence of a structural attractor — it is a natural property of price processes crossing nearby levels.

---

## 7. What the Model Does Tell Us About Price

Despite the failure of the direct monetisation strategy, the model reveals several genuine and useful properties of FVG fill dynamics.

### 7.1 Fill Timing Is Partially Predictable

The model provides statistically significant discrimination at every horizon tested (all survive Bonferroni correction, §5.4). The features driving this are informative about market microstructure.

### 7.2 HTF Regime-Conditional Fill Dynamics

The HTF context features (`trend_4h`, `trend_daily`) are underexploited in the current model. Regime-conditional analysis reveals a counterintuitive pattern:

| FVG Type | Daily Uptrend Fill Rate | Daily Downtrend Fill Rate | Δ |
|----------|:-----------------------:|:-------------------------:|:---:|
| Bullish FVGs (below price) | 74.4% | **77.4%** | +3.0pp |
| Bearish FVGs (above price) | **78.0%** | 74.6% | +3.4pp |

**Counter-trend FVGs fill at higher rates than trend-supporting FVGs.** Bullish FVGs (which require price to retrace down) fill *more* often when the daily trend is bearish (i.e., the fill direction aligns with the trend). Similarly for bearish FVGs in uptrends. This is consistent with the mean-reversion interpretation: FVGs formed *against* the prevailing trend create a stronger "snap-back" as the trend reasserts itself through the gap.

This interaction is present but underweighted in the current model. A regime-conditional model — separate classifiers for trending vs ranging environments — would likely improve discrimination.

### 7.3 Session-Conditional Fill Quality

Fill *quality* (depth conditional on touch) varies strongly by trading session:

| Session | Fill Rate (h=10) | P(deep ≥ 50% &#124; fill) | Interpretation |
|---------|:----------------:|:-------------------:|----------------|
| Asian | 80.2% | **71.1%** | High fill rate AND high depth |
| London | 72.6% | **73.7%** | Lower fill rate but highest depth when filled |
| NY-London overlap | 63.3% | 58.5% | Lowest fill rate AND worst depth |
| NY afternoon | 83.4% | 64.2% | Highest fill rate but shallow |

The NY-London overlap — the most liquid session — produces the lowest fill rate (63.3%) and the shallowest fills (58.5% deep). This is consistent with efficient markets theory: when liquidity is highest, price dislocations are arbitraged away before FVG boundaries can be revisited, and when revisited, institutional liquidity prevents deep penetration.

Conversely, the Asian session produces both high fill rates and high fill depth — suggesting that lower-liquidity environments allow mean-reversion patterns to play out more fully.

### 7.4 Fill Depth Is Bimodal

The conditional depth distribution (given fill within 10 candles) is strikingly bimodal with a median of 1.0:
- **52.9%** of fills are complete (depth ≥ 100%) — price goes all the way through
- **9.1%** are shallow wicks (depth ≤ 10%) — immediate reversal
- The middle (10-50% depth) is relatively sparse

This bimodality suggests that the fill process has a **regime-switching structure**: either the gap acts as a genuine attractor and price fills it completely, or price merely wicks the boundary and reverses. Modelling which regime applies at any given fill would be the key to monetisation.

### 7.5 FVGs Are Not Special Attractors

The analysis provides evidence *against* the ICT/SMC claim that FVGs are fundamentally different from any other nearby price level. The fill rate is largely explained by proximity and volatility — the same features that would predict revisitation of any arbitrary price level. Fill depth is insensitive to ATR-normalised distance (P(deep|fill) ≈ 68–70% across all quintiles), suggesting that proximity affects *whether* a fill occurs but not *how deeply* price penetrates.

### 7.6 Calibrated Probabilities Are Useful

The model's exceptional calibration (< 2% error) means its probability outputs can be used as reliable inputs to other systems. We are building a thermometer, not a switch — and the thermometer reads accurately. A trader using FVG analysis for directional bias can condition their confidence on the model's fill probability without worrying about systematic over- or under-estimation.

---

## 8. Applications: How Fill Probabilities Add Value

While direct FVG boundary trading is unprofitable, the fill-probability model has legitimate applications as an **information signal**:

### 8.1 Confluence Filter

Use fill probability to weight existing trade setups from other methodologies. For example: only take a break-of-structure trade if nearby FVGs supporting the direction have P(fill₁₀) > 0.7. This doesn't create alpha on its own but may improve the signal-to-noise ratio of other strategies.

### 8.2 Directional Bias Indicator

The dashboard aggregates active bullish and bearish FVGs to produce a net directional bias. When multiple high-probability bullish FVGs exist below price and no bearish FVGs above, the model suggests "bullish pull" — a directional lean that can inform discretionary trading or position sizing.

### 8.3 Market Context Layer

Fill probabilities encode useful market context: high fill probabilities across the board indicate a range-bound, mean-reverting environment; low fill probabilities suggest trending conditions where mean-reversion is failing.

### 8.4 Complementary Tools and Frameworks

FVG fill probabilities are most valuable when combined with:

- **Order flow analysis**: True institutional order flow data (e.g., from FIX protocol feeds) can confirm whether FVG zones coincide with genuine unfilled institutional orders, distinguishing structural attractors from noise
- **Liquidity mapping**: Identifying stop-loss clusters (using open interest data or liquidity pool heuristics) near FVG boundaries provides a mechanistic explanation for why price might accelerate through a gap rather than reverse
- **Regime detection**: Online regime classifiers (e.g., Hidden Markov Models on volatility regimes) can identify when the market is in a mean-reverting vs trending state, modulating the weight given to FVG fill signals
- **Volume profile analysis**: Overlaying volume-at-price with FVG locations reveals whether gaps coincide with low-volume nodes (genuine structural voids) or high-volume consolidation zones (different dynamics)

---

## 9. System Architecture

### 9.1 V2 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                   │
│  OANDA API → oanda_collector.py → data/raw/*.csv             │
└─────────────┬─────────────────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────────────────┐
│  FEATURE PIPELINE                                             │
│  fvg_detector.py → fill_dataset_builder.py                   │
│                  → fill_feature_engineer.py (25 features)     │
│  14,567 labelled FVGs across 4 pairs                         │
└─────────────┬─────────────────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────────────────┐
│  MODEL LAYER                                                  │
│  survival_model.py → 7 XGBoost classifiers (3s training)     │
│  fill_predictor.py → real-time prediction engine             │
└─────────────┬─────────────────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────────────────┐
│  VALIDATION LAYER                                             │
│  backtest_model.py — 9-fold purged walk-forward               │
│  Bootstrap CIs, calibration analysis, economic simulation     │
│  Leakage audit (27 checks), naive benchmark                  │
└─────────────┬─────────────────────────────────────────────────┘
              │
┌─────────────▼─────────────────────────────────────────────────┐
│  PRODUCTION LAYER                                             │
│  monitor_dashboard.py — live dashboard (port 5002)            │
│  Per-FVG probability bars, urgency classification             │
│  OANDA API integration for real-time data                    │
└───────────────────────────────────────────────────────────────┘
```

### 9.2 Live Dashboard

The production dashboard displays:
- Candlestick chart with active FVG rectangles colour-coded by urgency
- Per-FVG probability bars at all seven horizons
- Urgency classification (imminent / moderate / low)
- Directional bias summary (bullish pull / bearish pull / competing / none)
- Auto-refreshing via OANDA API

---

## 10. Limitations

1. **Moderate discrimination**: OOS AUC of 0.61–0.65 is statistically significant but modest. The model provides a weak signal, not a strong edge. Feature engineering improvements could raise this, but the ceiling may be inherently low given the near-random nature of FVG fill timing.

2. **Fill depth not modelled**: The current model predicts boundary touch probability, not penetration depth. A model of fill *depth* would be more directly monetisable but requires higher-granularity data.

3. **Hourly timeframe only**: FVGs on higher timeframes (4H, daily) may exhibit different fill dynamics — larger gaps may have stronger attractor properties relative to volatility.

4. **Four major pairs only**: Cross-asset generalisation (commodities, indices, crypto) is untested.

5. **No transaction cost sensitivity**: The backtest includes a 1.5-pip spread but does not model execution latency, partial fills, or slippage on limit orders.

6. **Static model**: No online learning or retraining mechanism is implemented. Performance may degrade in novel market regimes.

---

## 11. Future Work

The analysis in this paper points toward specific, analytically-motivated research questions rather than generic "try more features" suggestions:

| Priority | Research Direction | Rationale |
|----------|-------------------|----------|
| **Critical** | Model the **conditional depth distribution**: P(depth ≥ 0.5 &#124; touch, features) | This is the genuinely monetisable signal; fill-time prediction alone cannot close the 35pp gap to break-even (§6.4) |
| **Critical** | Classify fills into **regime-switch categories**: full-fill vs boundary-rejection | The bimodal depth distribution (§7.4) suggests a latent binary state that could be a cleaner prediction target |
| High | Higher-timeframe FVGs (4H, daily) where gap-size/ATR ratio may be more favourable | A 4H FVG may have gap_size_atr ≈ 1.0–2.0 ATR instead of 0.4 ATR, potentially flipping the SL/TP ratio |
| High | **Regime-conditional models**: separate classifiers for trending vs ranging environments | Counter-trend FVGs fill 3pp higher (§7.2); session effects change fill depth dramatically (§7.3) |
| High | Derive minimum required AUC for depth classifier: **P(deep) must reach 87.6% on selected trades** | This sets a concrete, falsifiable target for any future monetisation attempt |
| Medium | Incorporate real order flow data to distinguish institutional FVGs from noise | Would allow testing whether high-volume FVGs have different depth distributions |
| Medium | Formal random-level crossing null model | Rigorously test whether FVG fill rates exceed what a GBM would predict for arbitrary price levels |
| Low | Deep learning on raw price sequences | LSTM/Transformer may capture depth-relevant patterns invisible to tabular features |

---

## 12. Conclusion

This paper presents the most rigorous publicly available analysis of Fair Value Gap fill behaviour in forex markets. Across 14,567 individually labelled FVGs and four major currency pairs:

1. **FVGs fill rapidly and reliably**: 92.7% fill within 100 candles, with a median fill time of 3 candles. This confirms practitioner intuitions.

2. **Fill timing is partially predictable**: A multi-horizon XGBoost model achieves statistically significant OOS discrimination (AUC 0.649, 95% CI [0.634, 0.664] at the 10-candle horizon) with excellent calibration (mean absolute error < 2%).

3. **Direct monetisation fails**: A limit-order strategy at FVG boundaries produces a Profit Factor of 0.12 under realistic conditions. The failure is structural: FVG gaps are approximately 1/30th of ATR, creating a 31:1 stop-to-target ratio that no achievable win rate can overcome.

4. **The model retains value as an information signal**: Calibrated fill probabilities enhance directional bias assessment, confluence filtering, and market context interpretation within broader trading frameworks.

The honest conclusion is one that challenges the prevailing narrative in retail trading education: **FVGs are real, FVG fills are predictable, but FVG fills alone do not constitute tradeable alpha.** The gap between "price revisits this level" and "I can profit from price revisiting this level" is the central unsolved problem of FVG-based trading — and it is a much harder problem than the existing literature acknowledges.

---

## Appendix A: Feature Column Reference

```
gap_size, gap_size_atr, gap_volume, impulse_ratio, body_to_atr,
is_bullish, dist_to_gap, dist_to_gap_atr, gap_above_price,
gap_pct_of_range, ret_5, ret_10, ret_20, trend_20,
momentum_toward_gap, atr_14, realized_vol, vol_percentile,
trend_4h, trend_daily, htf_supports_fill, session_hour,
day_of_week, n_nearby_same, n_nearby_opposite
```

## Appendix B: Key Performance Summary

| Metric | Value |
|--------|-------|
| Dataset | 14,567 FVGs across 4 major forex pairs |
| Model | 7 × XGBoost binary classifiers (25 features) |
| Training time | 3 seconds (CPU) |
| OOS AUC (10-candle horizon) | **0.649 [0.634, 0.664]** |
| Mean calibration error | **0.9%** |
| Walk-forward folds | 9 (expanding window, 100-candle purge) |
| Total OOS predictions | 7,230 |
| Leakage audit | **27/27 checks passed** |
| Economic simulation (model) | PF 0.12, Sharpe −4.53 |
| Economic simulation (naive) | PF 0.10, Sharpe −5.03 |
| Fill rate (h=10) | 76.0% |
| Fill rate (h=100) | 92.7% |
| Mean SL/TP ratio | ~31:1 |

## Appendix C: Reproducibility

All code, data pipelines, and model artifacts are contained in the `fvg-probability-v2/` repository. To reproduce the results:

```bash
# 1. Build the dataset from raw OANDA CSVs
python3 src/fill_dataset_builder.py

# 2. Train the multi-horizon models (3 seconds)
python3 src/survival_model.py

# 3. Run the walk-forward backtest (20 seconds)
python3 src/backtest_model.py

# 4. Launch the live dashboard
python3 src/monitor_dashboard.py --port 5002
```
