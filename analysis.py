import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

PALETTE = {
    "Extreme Fear": "#d62728",
    "Fear":         "#ff7f0e",
    "Neutral":      "#bcbd22",
    "Greed":        "#2ca02c",
    "Extreme Greed":"#1f77b4",
}
SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENT_COLORS = [PALETTE[s] for s in SENT_ORDER]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

OUT = "/home/claude/bitcoin_sentiment_analysis/outputs/figures/"

import os
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("SECTION 1 — DATA UNDERSTANDING")
print("=" * 70)

trades_raw = pd.read_csv("/mnt/user-data/uploads/1779095246895_historical_data.csv")
fg_raw     = pd.read_csv("/mnt/user-data/uploads/1779096393647_fear_greed_index.csv")

print(f"\nTrades dataset  : {trades_raw.shape[0]:,} rows × {trades_raw.shape[1]} columns")
print(f"Fear & Greed    : {fg_raw.shape[0]:,} rows × {fg_raw.shape[1]} columns")
print(f"Unique traders  : {trades_raw['Account'].nunique()}")
print(f"Unique assets   : {trades_raw['Coin'].nunique()}")
print(f"\nTrades columns  : {trades_raw.columns.tolist()}")
print(f"FG columns      : {fg_raw.columns.tolist()}")

print("\n--- Trades null counts ---")
print(trades_raw.isnull().sum())

print("\n--- Direction distribution ---")
print(trades_raw['Direction'].value_counts())

print("\n--- Closed PnL (non-zero) ---")
nonzero_pnl = trades_raw[trades_raw['Closed PnL'] != 0]['Closed PnL']
print(nonzero_pnl.describe().round(2))

print("\n--- Fear & Greed classification counts ---")
print(fg_raw['classification'].value_counts())

print("\n" + "=" * 70)
print("SECTION 2 — DATA CLEANING & PREPROCESSING")
print("=" * 70)

trades = trades_raw.copy()
fg     = fg_raw.copy()

trades.columns = (trades.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_'))

fg.columns = fg.columns.str.strip().str.lower()

trades['timestamp_ist'] = pd.to_datetime(
    trades['timestamp_ist'], format='%d-%m-%Y %H:%M', errors='coerce')

fg['date'] = pd.to_datetime(fg['date'], errors='coerce')

trades['trade_date']  = trades['timestamp_ist'].dt.date
trades['trade_date']  = pd.to_datetime(trades['trade_date'])
trades['trading_hour']= trades['timestamp_ist'].dt.hour
trades['trading_day'] = trades['timestamp_ist'].dt.day_name()
trades['trade_month'] = trades['timestamp_ist'].dt.to_period('M').astype(str)

trades = trades.drop_duplicates(subset=['transaction_hash', 'trade_id'], keep='first')
print(f"\nRows after dedup: {len(trades):,}")

trades['closed_pnl']      = pd.to_numeric(trades['closed_pnl'], errors='coerce').fillna(0)
trades['size_usd']        = pd.to_numeric(trades['size_usd'], errors='coerce').fillna(0)
trades['execution_price'] = pd.to_numeric(trades['execution_price'], errors='coerce')
trades['fee']             = pd.to_numeric(trades['fee'], errors='coerce').fillna(0)
trades['size_tokens']     = pd.to_numeric(trades['size_tokens'], errors='coerce').fillna(0)
trades['start_position']  = pd.to_numeric(trades['start_position'], errors='coerce').fillna(0)

trades['side']      = trades['side'].str.upper().str.strip()
trades['direction'] = trades['direction'].str.strip()
trades['coin']      = trades['coin'].str.strip()

fg['classification'] = fg['classification'].str.strip()

sentiment_map = {
    "Extreme Fear": 1,
    "Fear":         2,
    "Neutral":      3,
    "Greed":        4,
    "Extreme Greed":5,
}
fg['sentiment_score'] = fg['classification'].map(sentiment_map)

trades['is_profitable']  = (trades['closed_pnl'] > 0).astype(int)
trades['has_closed_pnl'] = (trades['closed_pnl'] != 0).astype(int)
trades['net_pnl']        = trades['closed_pnl'] - trades['fee']
trades['is_opening']     = trades['direction'].isin(
    ['Open Long', 'Open Short', 'Buy']).astype(int)
trades['is_closing']     = trades['direction'].isin(
    ['Close Long', 'Close Short', 'Sell']).astype(int)

print("\nPreprocessing complete. Sample of cleaned trades:")
print(trades[['trade_date', 'coin', 'side', 'direction',
              'closed_pnl', 'net_pnl', 'size_usd']].head(5))

print("\n" + "=" * 70)
print("SECTION 3 — DATA MERGING")
print("=" * 70)

fg_merge = fg[['date', 'classification', 'sentiment_score', 'value']].copy()
fg_merge.columns = ['trade_date', 'sentiment', 'sentiment_score', 'fg_value']

merged = trades.merge(fg_merge, on='trade_date', how='left')

unmatched = merged['sentiment'].isna().sum()
print(f"\nTotal trades         : {len(merged):,}")
print(f"Matched with FG Index: {merged['sentiment'].notna().sum():,}")
print(f"Unmatched trades     : {unmatched:,}")

merged = merged.dropna(subset=['sentiment'])
merged['sentiment'] = pd.Categorical(
    merged['sentiment'], categories=SENT_ORDER, ordered=True)

print(f"\nFinal merged dataset : {len(merged):,} rows")
print(f"Date range           : {merged['trade_date'].min().date()} → "
      f"{merged['trade_date'].max().date()}")
print(f"\nTrades per sentiment class:")
print(merged['sentiment'].value_counts().reindex(SENT_ORDER))

print("\n" + "=" * 70)
print("SECTION 4 — FEATURE ENGINEERING")
print("=" * 70)

merged['risk_score'] = merged['size_usd'].abs() * merged['fg_value'] / 100

trader_stats = (merged.groupby('account')
                .agg(
                    total_trades  = ('closed_pnl', 'count'),
                    total_pnl     = ('closed_pnl', 'sum'),
                    win_trades    = ('is_profitable', 'sum'),
                    avg_pnl       = ('closed_pnl', 'mean'),
                    pnl_std       = ('closed_pnl', 'std'),
                    avg_size_usd  = ('size_usd', 'mean'),
                    total_fees    = ('fee', 'sum'),
                    avg_risk      = ('risk_score', 'mean'),
                )
                .reset_index())

trader_stats['win_rate']           = trader_stats['win_trades'] / trader_stats['total_trades']
trader_stats['sharpe_proxy']       = (trader_stats['avg_pnl'] /
                                       trader_stats['pnl_std'].replace(0, np.nan))
trader_stats['activity_score']     = np.log1p(trader_stats['total_trades'])
trader_stats['net_pnl_after_fees'] = trader_stats['total_pnl'] - trader_stats['total_fees']

merged = merged.merge(
    trader_stats[['account', 'win_rate', 'sharpe_proxy',
                  'activity_score', 'total_pnl', 'net_pnl_after_fees']],
    on='account', how='left')

merged['high_risk_trade']  = (merged['risk_score'] > merged['risk_score'].quantile(0.90)).astype(int)
merged['large_loss']       = (merged['closed_pnl'] < merged['closed_pnl'].quantile(0.05)).astype(int)
merged['large_win']        = (merged['closed_pnl'] > merged['closed_pnl'].quantile(0.95)).astype(int)

print("Feature engineering complete.")
print(f"Engineered columns added: risk_score, win_rate, sharpe_proxy, "
      f"activity_score, high_risk_trade, large_loss, large_win")

print("\n--- Trader-level stats (top 10 by total PnL) ---")
print(trader_stats.sort_values('total_pnl', ascending=False).head(10)
      [['account', 'total_trades', 'total_pnl', 'win_rate',
        'sharpe_proxy']].to_string(index=False))

print("\n" + "=" * 70)
print("SECTION 5 — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

closing_trades = merged[merged['closed_pnl'] != 0].copy()

print(f"\nClosing trades (non-zero PnL) : {len(closing_trades):,}")
print(f"Win rate overall              : {closing_trades['is_profitable'].mean():.1%}")
print(f"Avg closed PnL                : ${closing_trades['closed_pnl'].mean():.2f}")
print(f"Median closed PnL             : ${closing_trades['closed_pnl'].median():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pnl_clipped = closing_trades['closed_pnl'].clip(-2000, 5000)
axes[0].hist(pnl_clipped[closing_trades['closed_pnl'] < 0],
             bins=80, color='#d62728', alpha=0.7, label='Losses')
axes[0].hist(pnl_clipped[closing_trades['closed_pnl'] > 0],
             bins=80, color='#2ca02c', alpha=0.7, label='Profits')
axes[0].axvline(0, color='black', lw=1.5, ls='--')
axes[0].set_xlabel("Closed PnL (USD, clipped at ±$2000/$5000)")
axes[0].set_ylabel("Number of Trades")
axes[0].set_title("PnL Distribution — Profits vs Losses")
axes[0].legend()
axes[0].text(0.98, 0.95,
             f"Win rate: {closing_trades['is_profitable'].mean():.1%}",
             transform=axes[0].transAxes, ha='right', va='top',
             fontsize=10, color='#333333')

log_pnl = np.log1p(closing_trades.loc[closing_trades['closed_pnl'] > 0, 'closed_pnl'])
axes[1].hist(log_pnl, bins=60, color='#1f77b4', edgecolor='white', alpha=0.85)
axes[1].set_xlabel("log(1 + PnL) for profitable trades")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Log-Scale Profit Distribution (Right-Skewed Reality)")

plt.suptitle("Fig 1 — PnL Distribution Analysis", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig01_pnl_distribution.png", bbox_inches='tight')
plt.close()
print("\n[Saved] Fig 1 — PnL Distribution")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sent_pnl = (closing_trades.groupby('sentiment', observed=True)['closed_pnl']
            .agg(['mean', 'median', 'std']).reindex(SENT_ORDER))
bars = axes[0].bar(SENT_ORDER, sent_pnl['mean'],
                   color=SENT_COLORS, edgecolor='white', linewidth=0.8)
axes[0].axhline(0, color='black', lw=1, ls='--')
axes[0].set_xlabel("Market Sentiment")
axes[0].set_ylabel("Average Closed PnL (USD)")
axes[0].set_title("Average PnL by Market Sentiment")
axes[0].set_xticklabels(SENT_ORDER, rotation=20, ha='right')
for bar, val in zip(bars, sent_pnl['mean']):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"${val:.1f}", ha='center', va='bottom', fontsize=8)

win_rates = (closing_trades.groupby('sentiment', observed=True)['is_profitable']
             .mean().reindex(SENT_ORDER) * 100)
axes[1].bar(SENT_ORDER, win_rates, color=SENT_COLORS,
            edgecolor='white', linewidth=0.8)
axes[1].axhline(50, color='grey', lw=1.2, ls='--', label='50% baseline')
axes[1].set_xlabel("Market Sentiment")
axes[1].set_ylabel("Win Rate (%)")
axes[1].set_title("Win Rate by Market Sentiment")
axes[1].set_xticklabels(SENT_ORDER, rotation=20, ha='right')
axes[1].legend()

plt.suptitle("Fig 2 — Sentiment-wise Profitability", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig02_sentiment_profitability.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 2 — Sentiment Profitability")

fig, ax = plt.subplots(figsize=(12, 5))
pnl_clip = closing_trades.copy()
pnl_clip['pnl_clipped'] = pnl_clip['closed_pnl'].clip(-3000, 8000)
sns.boxplot(data=pnl_clip, x='sentiment', y='pnl_clipped',
            order=SENT_ORDER, palette=PALETTE, ax=ax,
            showfliers=False, linewidth=1.2)
ax.axhline(0, color='black', lw=1, ls='--', alpha=0.6)
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Closed PnL (USD, clipped ±$3000/8000)")
ax.set_title("Fig 3 — PnL Spread by Sentiment (Boxplot — Outliers Hidden)")
plt.tight_layout()
plt.savefig(OUT + "fig03_pnl_boxplot_sentiment.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 3 — PnL Boxplot")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

trade_vol = merged.groupby('sentiment', observed=True).size().reindex(SENT_ORDER)
axes[0].bar(SENT_ORDER, trade_vol, color=SENT_COLORS, edgecolor='white')
axes[0].set_xlabel("Market Sentiment")
axes[0].set_ylabel("Number of Trades")
axes[0].set_title("Trade Volume by Sentiment")
axes[0].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

avg_size = merged.groupby('sentiment', observed=True)['size_usd'].median().reindex(SENT_ORDER)
axes[1].bar(SENT_ORDER, avg_size, color=SENT_COLORS, edgecolor='white')
axes[1].set_xlabel("Market Sentiment")
axes[1].set_ylabel("Median Trade Size (USD)")
axes[1].set_title("Median Position Size by Sentiment")
axes[1].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

plt.suptitle("Fig 4 — Trading Activity by Sentiment", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig04_trade_volume_sentiment.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 4 — Trade Volume by Sentiment")

fig, ax = plt.subplots(figsize=(12, 5))
buy_sell = (merged.groupby(['sentiment', 'side'], observed=True)
            .size().unstack(fill_value=0)
            .reindex(SENT_ORDER))
buy_sell_pct = buy_sell.div(buy_sell.sum(axis=1), axis=0) * 100
buy_sell_pct[['BUY', 'SELL']].plot(
    kind='bar', stacked=True, ax=ax,
    color=['#2ca02c', '#d62728'], edgecolor='white')
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Percentage of Trades (%)")
ax.set_title("Fig 5 — Buy vs Sell Distribution by Sentiment")
ax.set_xticklabels(SENT_ORDER, rotation=20, ha='right')
ax.legend(['BUY', 'SELL'], loc='upper right')
plt.tight_layout()
plt.savefig(OUT + "fig05_buy_sell_sentiment.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 5 — Buy vs Sell")

daily_activity = (merged.groupby('trade_date')
                  .agg(trade_count=('closed_pnl', 'count'),
                       daily_pnl=('closed_pnl', 'sum'),
                       sentiment_score=('sentiment_score', 'first'),
                       sentiment=('sentiment', 'first'))
                  .reset_index())

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(daily_activity['trade_date'], daily_activity['trade_count'],
             color='#1f77b4', lw=1.2, alpha=0.9)
axes[0].fill_between(daily_activity['trade_date'],
                     daily_activity['trade_count'], alpha=0.15, color='#1f77b4')
axes[0].set_ylabel("Daily Trade Count")
axes[0].set_title("Daily Trade Activity")

axes[1].plot(daily_activity['trade_date'], daily_activity['daily_pnl'],
             color='#2ca02c', lw=1.2, alpha=0.9)
axes[1].axhline(0, color='black', lw=1, ls='--')
axes[1].fill_between(daily_activity['trade_date'],
                     daily_activity['daily_pnl'],
                     where=daily_activity['daily_pnl'] >= 0,
                     alpha=0.2, color='#2ca02c')
axes[1].fill_between(daily_activity['trade_date'],
                     daily_activity['daily_pnl'],
                     where=daily_activity['daily_pnl'] < 0,
                     alpha=0.2, color='#d62728')
axes[1].set_ylabel("Daily Cumulative PnL (USD)")
axes[1].set_title("Daily Net PnL Across All Traders")

for sent, color in PALETTE.items():
    mask = daily_activity['sentiment'] == sent
    axes[2].bar(daily_activity.loc[mask, 'trade_date'],
                daily_activity.loc[mask, 'sentiment_score'],
                color=color, alpha=0.85, label=sent, width=1.2)
axes[2].set_ylabel("Sentiment Score (1–5)")
axes[2].set_title("Daily Market Sentiment (Fear & Greed)")
axes[2].legend(loc='upper left', fontsize=8, ncol=5)
axes[2].set_xlabel("Date")

plt.suptitle("Fig 6 — Time-Series: Trades, PnL & Sentiment",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig06_timeseries.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 6 — Time Series")

fig, ax = plt.subplots(figsize=(12, 5))
risk_by_sent = (merged.groupby('sentiment', observed=True)['risk_score']
                .median().reindex(SENT_ORDER))
ax.bar(SENT_ORDER, risk_by_sent, color=SENT_COLORS, edgecolor='white')
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Median Risk Score (Size × FG Value / 100)")
ax.set_title("Fig 7 — Risk-Taking by Sentiment")
ax.set_xticklabels(SENT_ORDER, rotation=20, ha='right')
plt.tight_layout()
plt.savefig(OUT + "fig07_risk_by_sentiment.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 7 — Risk Score by Sentiment")

hour_pnl = (closing_trades.groupby('trading_hour')['closed_pnl']
            .agg(['mean', 'count']).reset_index())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(hour_pnl['trading_hour'], hour_pnl['mean'],
            color='#1f77b4', edgecolor='white')
axes[0].axhline(0, color='black', lw=1, ls='--')
axes[0].set_xlabel("Hour of Day (IST)")
axes[0].set_ylabel("Average PnL (USD)")
axes[0].set_title("Avg PnL by Trading Hour")

axes[1].bar(hour_pnl['trading_hour'], hour_pnl['count'],
            color='#ff7f0e', edgecolor='white')
axes[1].set_xlabel("Hour of Day (IST)")
axes[1].set_ylabel("Trade Count")
axes[1].set_title("Trade Frequency by Hour")

plt.suptitle("Fig 8 — Intraday Trading Patterns (IST)", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig08_intraday.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 8 — Intraday")

top_coins = trades['coin'].value_counts().head(10).index.tolist()
coin_data = closing_trades[closing_trades['coin'].isin(top_coins)]
coin_pnl = (coin_data.groupby('coin')['closed_pnl']
            .agg(['mean', 'sum', 'count'])
            .sort_values('sum', ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(coin_pnl.index, coin_pnl['sum'],
             color=['#2ca02c' if x >= 0 else '#d62728'
                    for x in coin_pnl['sum']])
axes[0].axvline(0, color='black', lw=1)
axes[0].set_xlabel("Total Cumulative PnL (USD)")
axes[0].set_title("Total PnL by Asset (Top 10)")

axes[1].barh(coin_pnl.index, coin_pnl['mean'],
             color=['#2ca02c' if x >= 0 else '#d62728'
                    for x in coin_pnl['mean']])
axes[1].axvline(0, color='black', lw=1)
axes[1].set_xlabel("Mean PnL per Trade (USD)")
axes[1].set_title("Average PnL per Trade by Asset")

plt.suptitle("Fig 9 — Asset-Level Performance Analysis",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig09_asset_performance.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 9 — Asset Performance")

print("\n" + "=" * 70)
print("SECTION 6 — STATISTICAL ANALYSIS")
print("=" * 70)

print("\n--- Pearson Correlation: FG Score vs Closed PnL ---")
corr_data = closing_trades.dropna(subset=['fg_value', 'closed_pnl'])
r, p = pearsonr(corr_data['fg_value'], corr_data['closed_pnl'])
print(f"  r = {r:.4f}, p-value = {p:.4e}")
print(f"  Interpretation: {'Statistically significant' if p < 0.05 else 'Not significant'} "
      f"({'positive' if r > 0 else 'negative'}) correlation")

print("\n--- Correlation: Risk Score vs Closed PnL ---")
r2, p2 = pearsonr(corr_data['risk_score'].fillna(0), corr_data['closed_pnl'])
print(f"  r = {r2:.4f}, p-value = {p2:.4e}")

print("\n--- ANOVA: Does sentiment significantly impact PnL? ---")
groups = [closing_trades.loc[closing_trades['sentiment'] == s, 'closed_pnl'].values
          for s in SENT_ORDER
          if (closing_trades['sentiment'] == s).sum() > 0]
f_stat, p_anova = f_oneway(*groups)
print(f"  H0: Mean PnL is equal across all sentiment classes")
print(f"  H1: At least one sentiment class has different mean PnL")
print(f"  F-statistic = {f_stat:.4f}, p-value = {p_anova:.4e}")
print(f"  Result: {'REJECT H0 — sentiment significantly impacts PnL' if p_anova < 0.05 else 'FAIL TO REJECT H0'}")

print("\n--- T-test: Fear vs Greed PnL ---")
fear_pnl  = closing_trades.loc[closing_trades['sentiment'].isin(
    ['Extreme Fear', 'Fear']), 'closed_pnl']
greed_pnl = closing_trades.loc[closing_trades['sentiment'].isin(
    ['Greed', 'Extreme Greed']), 'closed_pnl']
t_stat, p_t = ttest_ind(fear_pnl, greed_pnl, equal_var=False)
print(f"  Fear  mean PnL : ${fear_pnl.mean():.2f}  (n={len(fear_pnl):,})")
print(f"  Greed mean PnL : ${greed_pnl.mean():.2f}  (n={len(greed_pnl):,})")
print(f"  t = {t_stat:.4f}, p = {p_t:.4e}")
print(f"  Result: {'Statistically significant difference' if p_t < 0.05 else 'No significant difference'}")

print("\n--- T-test: Risk Score in Extreme Greed vs Extreme Fear ---")
eg_risk = merged.loc[merged['sentiment'] == 'Extreme Greed', 'risk_score']
ef_risk = merged.loc[merged['sentiment'] == 'Extreme Fear',  'risk_score']
t3, p3  = ttest_ind(eg_risk.dropna(), ef_risk.dropna(), equal_var=False)
print(f"  Extreme Greed avg risk: {eg_risk.mean():.2f}")
print(f"  Extreme Fear  avg risk: {ef_risk.mean():.2f}")
print(f"  t = {t3:.4f}, p = {p3:.4e}")
print(f"  Result: {'Risk-taking significantly higher during Extreme Greed' if (t3 > 0 and p3 < 0.05) else 'No significant difference'}")

print("\n--- T-test: Win rate — Greed vs Fear ---")
greed_win = closing_trades.loc[closing_trades['sentiment'].isin(['Greed', 'Extreme Greed']), 'is_profitable']
fear_win  = closing_trades.loc[closing_trades['sentiment'].isin(['Fear', 'Extreme Fear']),   'is_profitable']
t4, p4   = ttest_ind(greed_win, fear_win, equal_var=False)
print(f"  Greed win rate: {greed_win.mean():.2%}")
print(f"  Fear  win rate: {fear_win.mean():.2%}")
print(f"  t = {t4:.4f}, p = {p4:.4e}")

num_cols = ['closed_pnl', 'size_usd', 'fee', 'risk_score',
            'fg_value', 'sentiment_score', 'is_profitable', 'net_pnl']
corr_mat = closing_trades[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Fig 10 — Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig(OUT + "fig10_correlation_matrix.png", bbox_inches='tight')
plt.close()
print("\n[Saved] Fig 10 — Correlation Matrix")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

pnl_std_by_sent = (closing_trades.groupby('sentiment', observed=True)['closed_pnl']
                   .std().reindex(SENT_ORDER))
axes[0, 0].bar(SENT_ORDER, pnl_std_by_sent, color=SENT_COLORS, edgecolor='white')
axes[0, 0].set_title("PnL Volatility (Std Dev) by Sentiment")
axes[0, 0].set_ylabel("PnL Std Dev (USD)")
axes[0, 0].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

high_risk_pct = (merged.groupby('sentiment', observed=True)['high_risk_trade']
                 .mean().reindex(SENT_ORDER) * 100)
axes[0, 1].bar(SENT_ORDER, high_risk_pct, color=SENT_COLORS, edgecolor='white')
axes[0, 1].set_title("% High-Risk Trades by Sentiment")
axes[0, 1].set_ylabel("High-Risk Trade Ratio (%)")
axes[0, 1].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

large_loss_pct = (closing_trades.groupby('sentiment', observed=True)['large_loss']
                  .mean().reindex(SENT_ORDER) * 100)
axes[1, 0].bar(SENT_ORDER, large_loss_pct, color=SENT_COLORS, edgecolor='white')
axes[1, 0].set_title("% Large Loss Trades by Sentiment (Bottom 5%)")
axes[1, 0].set_ylabel("Large Loss Trade Ratio (%)")
axes[1, 0].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

large_win_pct = (closing_trades.groupby('sentiment', observed=True)['large_win']
                 .mean().reindex(SENT_ORDER) * 100)
axes[1, 1].bar(SENT_ORDER, large_win_pct, color=SENT_COLORS, edgecolor='white')
axes[1, 1].set_title("% Large Win Trades by Sentiment (Top 5%)")
axes[1, 1].set_ylabel("Large Win Trade Ratio (%)")
axes[1, 1].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

plt.suptitle("Fig 11 — Risk Analysis by Sentiment",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig11_risk_analysis.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 11 — Risk Analysis")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

trader_rank = trader_stats.sort_values('total_pnl', ascending=False).head(15)
axes[0].barh(trader_rank['account'].str[:12], trader_rank['total_pnl'],
             color=['#2ca02c' if x >= 0 else '#d62728'
                    for x in trader_rank['total_pnl']])
axes[0].axvline(0, color='black', lw=1)
axes[0].set_xlabel("Total Cumulative PnL (USD)")
axes[0].set_title("Top 15 Traders by Cumulative PnL")
axes[0].invert_yaxis()

axes[1].scatter(trader_stats['total_trades'],
                trader_stats['total_pnl'],
                c=trader_stats['win_rate'],
                cmap='RdYlGn', alpha=0.75, s=60,
                edgecolors='grey', linewidth=0.3)
sm = plt.cm.ScalarMappable(cmap='RdYlGn',
     norm=plt.Normalize(vmin=trader_stats['win_rate'].min(),
                        vmax=trader_stats['win_rate'].max()))
plt.colorbar(sm, ax=axes[1], label='Win Rate')
axes[1].axhline(0, color='black', lw=1, ls='--')
axes[1].set_xlabel("Total Number of Trades")
axes[1].set_ylabel("Total Cumulative PnL (USD)")
axes[1].set_title("Trade Frequency vs Cumulative PnL\n(Color = Win Rate)")

plt.suptitle("Fig 12 — Trader-Level Performance",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig12_trader_performance.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 12 — Trader Performance")

top_accounts = trader_stats.nlargest(5, 'total_trades')['account'].tolist()
top_trader_data = merged[merged['account'].isin(top_accounts)].copy()
top_trader_data = top_trader_data.sort_values('trade_date')

fig, ax = plt.subplots(figsize=(14, 6))
for acct in top_accounts:
    sub = top_trader_data[top_trader_data['account'] == acct]
    cum_pnl = sub['closed_pnl'].cumsum()
    ax.plot(sub['trade_date'].values, cum_pnl.values,
            label=acct[:12], lw=1.5, alpha=0.8)
ax.axhline(0, color='black', lw=1, ls='--')
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative PnL (USD)")
ax.set_title("Fig 13 — Cumulative PnL Trajectories (Top 5 Most Active Traders)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "fig13_cumulative_pnl.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 13 — Cumulative PnL Trajectories")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_pnl   = (closing_trades.groupby('trading_day')['closed_pnl']
             .mean().reindex(day_order))
day_count = (closing_trades.groupby('trading_day').size().reindex(day_order))

axes[0].bar(day_order, day_pnl,
            color=['#2ca02c' if x >= 0 else '#d62728' for x in day_pnl],
            edgecolor='white')
axes[0].axhline(0, color='black', lw=1, ls='--')
axes[0].set_title("Avg PnL by Day of Week")
axes[0].set_ylabel("Avg PnL (USD)")
axes[0].set_xticklabels(day_order, rotation=30, ha='right')

axes[1].bar(day_order, day_count, color='#1f77b4', edgecolor='white')
axes[1].set_title("Trade Count by Day of Week")
axes[1].set_ylabel("Number of Trades")
axes[1].set_xticklabels(day_order, rotation=30, ha='right')

plt.suptitle("Fig 14 — Day-of-Week Patterns",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig14_day_of_week.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 14 — Day of Week")

coin_sent = (closing_trades[closing_trades['coin'].isin(top_coins)]
             .groupby(['coin', 'sentiment'], observed=True)['closed_pnl']
             .mean().unstack(fill_value=0)
             .reindex(columns=SENT_ORDER))

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(coin_sent, annot=True, fmt='.0f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax)
ax.set_title("Fig 15 — Mean PnL by Asset × Sentiment (Heatmap)")
ax.set_xlabel("Market Sentiment")
ax.set_ylabel("Asset")
plt.tight_layout()
plt.savefig(OUT + "fig15_asset_sentiment_heatmap.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 15 — Asset × Sentiment Heatmap")

print("\n" + "=" * 70)
print("SECTION 7 — BEHAVIORAL FINANCE INSIGHTS")
print("=" * 70)

print("\n--- Overconfidence during Extreme Greed ---")
eg_data = closing_trades[closing_trades['sentiment'] == 'Extreme Greed']
ef_data = closing_trades[closing_trades['sentiment'] == 'Extreme Fear']

if len(eg_data) > 0 and len(ef_data) > 0:
    print(f"  Extreme Greed: avg size ${eg_data['size_usd'].mean():,.0f}, "
          f"win rate {eg_data['is_profitable'].mean():.1%}")
    print(f"  Extreme Fear:  avg size ${ef_data['size_usd'].mean():,.0f}, "
          f"win rate {ef_data['is_profitable'].mean():.1%}")
    print(f"  Greed/Fear size ratio: "
          f"{eg_data['size_usd'].mean() / max(ef_data['size_usd'].mean(), 0.01):.2f}x")

print("\n--- Direction Mix by Sentiment ---")
dir_sent = (merged.groupby(['sentiment', 'direction'], observed=True)
            .size().unstack(fill_value=0)
            .reindex(SENT_ORDER))
long_cols  = [c for c in dir_sent.columns if 'Long' in c or c == 'Buy']
short_cols = [c for c in dir_sent.columns if 'Short' in c or c == 'Sell']
dir_sent['long_count']  = dir_sent[long_cols].sum(axis=1)
dir_sent['short_count'] = dir_sent[short_cols].sum(axis=1)
dir_sent['long_pct']    = dir_sent['long_count'] / (
    dir_sent['long_count'] + dir_sent['short_count']) * 100
print(dir_sent[['long_count', 'short_count', 'long_pct']].to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(SENT_ORDER, dir_sent['long_pct'], color=SENT_COLORS, edgecolor='white')
axes[0].axhline(50, color='grey', lw=1.2, ls='--', label='50% baseline')
axes[0].set_ylabel("% Long-Biased Trades")
axes[0].set_title("Long Bias by Sentiment\n(Are traders bullish during greed?)")
axes[0].set_xticklabels(SENT_ORDER, rotation=20, ha='right')
axes[0].legend()

pnl_vol = (closing_trades.groupby('sentiment', observed=True)['closed_pnl']
           .std().reindex(SENT_ORDER))
axes[1].bar(SENT_ORDER, pnl_vol, color=SENT_COLORS, edgecolor='white')
axes[1].set_ylabel("PnL Std Dev (USD)")
axes[1].set_title("Emotional Volatility — PnL Spread\nby Sentiment Class")
axes[1].set_xticklabels(SENT_ORDER, rotation=20, ha='right')

plt.suptitle("Fig 16 — Behavioral Finance Insights",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig16_behavioral_insights.png", bbox_inches='tight')
plt.close()
print("\n[Saved] Fig 16 — Behavioral Insights")

print("\n" + "=" * 70)
print("SECTION 8 — PREDICTIVE MODELING")
print("=" * 70)

model_df = closing_trades[[
    'sentiment_score', 'fg_value', 'size_usd', 'fee',
    'risk_score', 'is_profitable', 'trading_hour',
    'is_opening', 'is_closing', 'net_pnl'
]].copy()

model_df['side_enc'] = (merged.loc[closing_trades.index, 'side']
                        .map({'BUY': 1, 'SELL': 0})
                        .fillna(0).values)

le_coin = LabelEncoder()
model_df['coin_enc'] = le_coin.fit_transform(
    merged.loc[closing_trades.index, 'coin'].fillna('UNK').values)

model_df = model_df.dropna()
X = model_df.drop('is_profitable', axis=1)
y = model_df['is_profitable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Class balance — Train: {y_train.mean():.1%} profitable")

rf_model = RandomForestClassifier(n_estimators=150, max_depth=8,
                                   min_samples_leaf=20,
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_auc    = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print("\n--- Random Forest Results ---")
print(classification_report(y_test, y_pred_rf))
print(f"AUC-ROC: {rf_auc:.4f}")

xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=5,
                                learning_rate=0.05, subsample=0.8,
                                use_label_encoder=False,
                                eval_metric='logloss', random_state=42,
                                verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_auc    = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print("\n--- XGBoost Results ---")
print(classification_report(y_test, y_pred_xgb))
print(f"AUC-ROC: {xgb_auc:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_rf, ax=axes[0],
    display_labels=['Loss', 'Profit'],
    colorbar=False, cmap='Blues')
axes[0].set_title(f"Random Forest\nAUC = {rf_auc:.3f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_xgb, ax=axes[1],
    display_labels=['Loss', 'Profit'],
    colorbar=False, cmap='Greens')
axes[1].set_title(f"XGBoost\nAUC = {xgb_auc:.3f}")

feat_imp = pd.Series(rf_model.feature_importances_,
                     index=X.columns).sort_values(ascending=True)
axes[2].barh(feat_imp.index, feat_imp.values, color='#1f77b4')
axes[2].set_title("Feature Importance (Random Forest)")
axes[2].set_xlabel("Importance Score")

plt.suptitle("Fig 17 — Predictive Model Results",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig17_model_results.png", bbox_inches='tight')
plt.close()
print("[Saved] Fig 17 — Model Results")

print("\n" + "=" * 70)
print("SECTION 9 — RISK ANALYSIS")
print("=" * 70)

print("\n--- Sharpe-like metric per trader ---")
print(trader_stats[['account', 'total_trades', 'total_pnl',
                     'win_rate', 'sharpe_proxy']]
      .sort_values('sharpe_proxy', ascending=False).head(10).to_string(index=False))

print("\n--- PnL volatility by sentiment (Std Dev) ---")
print(closing_trades.groupby('sentiment', observed=True)['closed_pnl']
      .std().reindex(SENT_ORDER).round(2))

print("\n--- High-risk trade concentration ---")
print(f"  90th percentile risk score threshold: "
      f"{merged['risk_score'].quantile(0.90):.2f}")
print(f"  High-risk trades: {merged['high_risk_trade'].sum():,} "
      f"({merged['high_risk_trade'].mean():.1%} of all trades)")
print(f"  High-risk trades by sentiment:")
print(merged.groupby('sentiment', observed=True)['high_risk_trade']
      .mean().reindex(SENT_ORDER).mul(100).round(1))

print("\n--- Extreme loss analysis ---")
bottom5 = closing_trades.nsmallest(50, 'closed_pnl')[
    ['account', 'coin', 'closed_pnl', 'size_usd', 'sentiment']]
print(bottom5.head(10).to_string(index=False))

daily_pnl_series = (closing_trades.groupby('trade_date')['closed_pnl'].sum())
cumulative        = daily_pnl_series.cumsum()
rolling_max       = cumulative.cummax()
drawdown          = cumulative - rolling_max

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(cumulative.index, cumulative.values, color='#2ca02c', lw=1.5)
axes[0].fill_between(cumulative.index, cumulative.values, alpha=0.15, color='#2ca02c')
axes[0].set_ylabel("Cumulative PnL (USD)")
axes[0].set_title("Cumulative PnL — All Traders Combined")

axes[1].fill_between(drawdown.index, drawdown.values, 0,
                     where=drawdown.values < 0, alpha=0.6, color='#d62728')
axes[1].set_ylabel("Drawdown (USD)")
axes[1].set_title("Portfolio-Level Drawdown")
axes[1].set_xlabel("Date")

plt.suptitle("Fig 18 — Cumulative PnL & Drawdown Analysis",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + "fig18_drawdown.png", bbox_inches='tight')
plt.close()
print("\n[Saved] Fig 18 — Drawdown Chart")

print("\n" + "=" * 70)
print("SECTION 10 — FINAL SUMMARY STATISTICS")
print("=" * 70)

total_volume   = merged['size_usd'].sum()
total_pnl_all  = closing_trades['closed_pnl'].sum()
overall_wr     = closing_trades['is_profitable'].mean()
avg_fee_paid   = merged['fee'].mean()
best_day_pnl   = daily_pnl_series.max()
worst_day_pnl  = daily_pnl_series.min()
max_dd         = drawdown.min()

print(f"""
  Total trades analyzed     : {len(merged):,}
  Unique traders            : {merged['account'].nunique()}
  Unique assets traded      : {merged['coin'].nunique()}
  Date range                : {merged['trade_date'].min().date()} → {merged['trade_date'].max().date()}
  Total USD volume          : ${total_volume:,.0f}
  Total cumulative PnL      : ${total_pnl_all:,.2f}
  Overall win rate          : {overall_wr:.2%}
  Average fee per trade     : ${avg_fee_paid:.4f}
  Best single day PnL       : ${best_day_pnl:,.2f}
  Worst single day PnL      : ${worst_day_pnl:,.2f}
  Max drawdown              : ${max_dd:,.2f}
  RF Model AUC-ROC          : {rf_auc:.4f}
  XGBoost AUC-ROC           : {xgb_auc:.4f}
""")

print("\n" + "=" * 70)
print("ALL FIGURES SAVED TO:", OUT)
print("=" * 70)
