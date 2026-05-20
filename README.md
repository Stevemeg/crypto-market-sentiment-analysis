# Bitcoin Market Sentiment & Trader Performance Analysis
### Hyperliquid × Fear & Greed Index — Quantitative Research Project

---

## Overview

This project investigates whether Bitcoin market sentiment — as measured by the Crypto Fear & Greed Index — meaningfully influences trader behavior, risk-taking patterns, and profitability on Hyperliquid, a high-performance perpetual DEX.

The analysis spans **May 2023 to May 2025**, covering **201,113 trades** across **32 wallet addresses** and **231 unique assets**, with a total notional volume exceeding **$1.15 billion USD**.

---

## Key Findings

| Finding | Evidence |
|---|---|
| Sentiment significantly impacts PnL | ANOVA: F=7.58, p<0.00001 |
| Risk-taking is 2.45× higher during Extreme Greed | T-test: t=27.36, p<10⁻¹⁶⁰ |
| Fear periods produce higher win rates | Fear: 84.2% vs Greed: 81.7%, p<10⁻²¹ |
| Extreme sentiment amplifies PnL variance | EF std: $1,656 vs Neutral: $758 |
| Traders are contrarian long during Fear | 65.7% long bias in Extreme Fear |
| High-risk trades peak at Greed (13.3%) | vs 5.1% in Extreme Fear |

---

## Project Structure

```
bitcoin_sentiment_analysis/
├── data/
│   ├── historical_data.csv          # Hyperliquid trade history
│   └── fear_greed_index.csv         # Daily BTC Fear & Greed readings
│
├── notebooks/
│   └── bitcoin_sentiment_analysis.ipynb   # Full analysis notebook
│
├── outputs/
│   └── figures/                     # All saved visualizations
│       ├── fig01_pnl_distribution.png
│       ├── fig02_sentiment_profitability.png
│       ├── fig03_pnl_boxplot_sentiment.png
│       ├── fig04_trade_volume_sentiment.png
│       ├── fig05_buy_sell_sentiment.png
│       ├── fig06_timeseries.png
│       ├── fig07_risk_by_sentiment.png
│       ├── fig08_intraday.png
│       ├── fig09_asset_performance.png
│       ├── fig10_correlation_matrix.png
│       ├── fig11_risk_analysis.png
│       ├── fig12_trader_performance.png
│       ├── fig13_cumulative_pnl.png
│       ├── fig14_day_of_week.png
│       ├── fig15_asset_sentiment_heatmap.png
│       ├── fig16_behavioral_insights.png
│       ├── fig17_model_results.png
│       └── fig18_drawdown.png
│
├── analysis.py                      # Standalone Python script (full pipeline)
├── requirements.txt
└── README.md
```

---

## Setup & Running

```bash
git clone <your-repo-url>
cd bitcoin_sentiment_analysis

pip install -r requirements.txt

cp /path/to/historical_data.csv data/
cp /path/to/fear_greed_index.csv data/

python analysis.py

jupyter notebook notebooks/bitcoin_sentiment_analysis.ipynb
```

---

## Analysis Pipeline

```
Raw Data
  │
  ├── Data Understanding (shape, nulls, dtypes, distributions)
  │
  ├── Cleaning & Preprocessing
  │     ├── Timestamp parsing (IST format)
  │     ├── Deduplication (-10,105 rows)
  │     ├── Numeric type enforcement
  │     └── Derived columns (trade_date, hour, day, net_pnl, flags)
  │
  ├── Merging (left join on trade_date)
  │     └── 201,113 rows matched (99.997%)
  │
  ├── Feature Engineering
  │     ├── risk_score = size_usd × fg_value / 100
  │     ├── win_rate, sharpe_proxy (per trader)
  │     ├── activity_score = log1p(total_trades)
  │     └── high_risk_trade, large_loss, large_win (binary flags)
  │
  ├── EDA (18 charts)
  │     ├── PnL distributions
  │     ├── Sentiment-wise profitability & win rates
  │     ├── Trade volume, size, buy/sell by sentiment
  │     ├── Time-series (daily activity, PnL, sentiment)
  │     ├── Intraday patterns
  │     ├── Asset-level performance
  │     ├── Trader-level performance & cumulative PnL
  │     ├── Day-of-week patterns
  │     └── Asset × Sentiment heatmap
  │
  ├── Statistical Testing
  │     ├── ANOVA: sentiment vs PnL (F=7.58, p<0.00001)
  │     ├── Pearson: FG score vs PnL (r=0.01, p=0.002)
  │     ├── T-test: risk (EG vs EF) — t=27.36, p<10⁻¹⁶⁰
  │     └── T-test: win rate (Greed vs Fear) — p<10⁻²¹
  │
  ├── Behavioral Finance
  │     ├── Long bias by sentiment (contrarian detection)
  │     └── PnL volatility by emotional state
  │
  ├── Risk Analysis
  │     ├── Sharpe-like metric per trader
  │     ├── High-risk trade concentration
  │     ├── Extreme loss clustering
  │     └── Cumulative PnL & drawdown
  │
  └── Predictive Modeling
        ├── Random Forest (AUC: 1.000)
        ├── XGBoost (AUC: 1.000)
        └── Feature Importance
```

---

## Statistical Results Summary

### ANOVA — Sentiment vs PnL
- **H₀:** Mean PnL equal across all sentiment classes
- **Result:** F = 7.58, p = 4.2 × 10⁻⁶ → **Reject H₀**

### T-test — Risk During Extreme Greed vs Extreme Fear
- **Result:** t = 27.36, p < 10⁻¹⁶⁰ → Greed risk is **2.45× higher**

### T-test — Win Rate: Fear vs Greed
- **Result:** Fear: 84.2%, Greed: 81.7%, p < 10⁻²¹ → **Fear wins more often**

---

## Strategic Recommendations

1. **Use FG score as a position-sizing lever** — scale down exposure as FG > 80, increase at FG < 25
2. **Deploy contrarian long strategies during Extreme Fear** — statistically validated higher win rates
3. **Implement hard leverage limits during Extreme Greed** — overconfidence bias is measurable and damaging
4. **Optimize for maker orders** — fee drag across 200K+ trades is a significant performance leak
5. **Monitor HYPE and BTC dominance** — two assets account for nearly 50% of all volume

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
scipy>=1.11.0
plotly>=5.15.0
jupyter>=1.0.0
```

---

## Author Notes

This project was built as part of a quantitative research internship assignment. The methodology follows institutional-grade EDA practices, including:
- Statistical hypothesis testing with explicit H₀/H₁ formulation
- Behavioral finance framework (overconfidence bias, loss aversion, herding)
- Risk-adjusted performance metrics (Sharpe proxy, drawdown analysis)
- Predictive modeling with proper train/test stratification
