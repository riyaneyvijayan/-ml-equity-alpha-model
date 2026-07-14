# Machine Learning Driven Multi-Factor Equity Alpha Model

An independent project building a systematic equity selection model using momentum, volatility, and price-based factors, evaluated with a supervised learning framework and backtested against a naive benchmark.

## Overview

This project investigates whether a small set of classic equity factors — momentum, volatility, and price relative to a long-term moving average — can be combined through a machine learning classifier to predict short-term stock outperformance, and whether acting on those predictions improves risk-adjusted returns versus a simple equal-weighted buy-and-hold approach.

## Data

- **Universe:** 31 large-cap US equities spanning technology, financials, energy, healthcare, consumer staples, consumer discretionary, industrials, and utilities
- **Period:** January 2020 – December 2025, daily closing prices
- **Source:** Yahoo Finance (via the `yfinance` Python library)

## Methodology

### Feature engineering

Three factors were constructed for each stock, at each point in time:

| Factor | Definition | Rationale |
|---|---|---|
| Momentum | 126-day (≈6 month) trailing return | Captures trend-following behaviour in equity returns |
| Volatility | 21-day (≈1 month) rolling standard deviation of daily returns | Captures recent risk/uncertainty in the stock |
| Price factor | % distance of current price from its 200-day moving average | Captures longer-term trend positioning |

### Labeling

The target variable is binary: whether a stock's forward 21-day (≈1 month) return is positive (1) or negative (0). Across the full dataset, the label distribution was approximately 56% positive / 44% negative — a reasonably balanced split, consistent with the broadly upward-trending 2020–2025 sample period.

### Train/test split

The dataset was split **chronologically** — the first 80% of the timeline for training (~32,400 rows), the most recent 20% (~8,100 rows) held out for testing. A random split was deliberately avoided, since shuffling time-series data would allow the model to implicitly "see the future" during training (lookahead bias).

### Models

Two classifiers were trained and compared:

1. **Logistic regression** (linear baseline)
2. **XGBoost** (gradient-boosted decision trees, `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`)

## Results

### Classification performance

| Model | Accuracy | Class 0 (down) Precision/Recall | Class 1 (up) Precision/Recall |
|---|---|---|---|
| Logistic Regression | 54.4% | 0.00 / 0.00 | 0.54 / 1.00 |
| XGBoost | 52.5% | 0.44 / 0.16 | 0.54 / 0.83 |

**Note on interpretation:** logistic regression's higher headline accuracy is misleading — it arose from the model defaulting to predicting the majority class ("up") for nearly every observation, which trivially matches the ~56% base rate. It captured no real signal, as reflected in its 0.00 precision/recall on the down-class. XGBoost's slightly lower accuracy came with genuine, if modest, discriminative power on both classes, making it the more meaningful and defensible model despite the lower top-line number.

### Feature importance (XGBoost)

| Feature | Importance |
|---|---|
| Momentum | 0.372 |
| Price factor | 0.339 |
| Volatility | 0.289 |

No single factor dominated; the model draws on a relatively even blend of all three, suggesting the predictive signal (such as it is) is distributed rather than concentrated in one dimension.

### Backtest (test period only, out-of-sample)

A simple long-only strategy was simulated: on each day, hold an equal-weighted portfolio of stocks the model predicted would rise; hold cash (0% return) for stocks predicted to fall. This was compared against an equal-weighted buy-and-hold benchmark of the full universe over the same period.

| Metric | ML Strategy | Benchmark |
|---|---|---|
| Annualized Sharpe ratio | 1.42 | 1.05 |
| Max drawdown | -13.64% | -16.77% |
| Final value (per $1 invested) | $1.20 | $1.18 |

The strategy modestly outperformed the benchmark in absolute terms, but the more notable result is the improvement in **risk-adjusted return** (higher Sharpe) and a **shallower maximum drawdown** — the model-driven approach reduced downside during the test period's volatility spike without sacrificing return.

## Limitations

- **Small feature set:** only three factors were used; no fundamental, sentiment, or macro data was incorporated.
- **Simple label:** forward 21-day binary direction is a coarse target; it doesn't account for magnitude of return or transaction costs.
- **No transaction costs or slippage:** the backtest assumes frictionless daily rebalancing, which would not hold in live trading.
- **Single test period:** results reflect one 2024–2025 out-of-sample window; performance is not validated across multiple market regimes (e.g. a sustained bear market).
- **Modest predictive power:** classification accuracy only modestly exceeds the base rate; this is a realistic and expected outcome for short-horizon equity direction prediction using simple technical factors, not a limitation unique to this implementation.

## Tools Used

Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, Matplotlib, Jupyter Notebook

## Possible Extensions

- Incorporate fundamental factors (valuation, quality, growth)
- Test alternative labeling schemes (e.g. return magnitude via regression rather than binary classification)
- Walk-forward / rolling-window validation across multiple market regimes
- Incorporate transaction costs and turnover constraints into the backtest
