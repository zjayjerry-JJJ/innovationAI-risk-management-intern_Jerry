import pandas as pd
import numpy as np


initial_capital = 1000000
transaction_cost = 0.001

log = []

# read data
prices = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)
weights = pd.read_csv('weights.csv', index_col=0, parse_dates=True)

log.append(" Step 1: Data Loaded")
log.append(f"Prices shape: {prices.shape}")
log.append(f"Weights shape: {weights.shape}")

# data match
weights = weights.reindex(prices.index).fillna(0)

log.append("📌 Step 2: Index aligned")
log.append(f"Index aligned? {prices.index.equals(weights.index)}")

# return match
returns = prices.pct_change().fillna(0)

# portfolio match
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

# turnover and cost
turnover = (weights - weights.shift(1)).abs().sum(axis=1)
cost = turnover * transaction_cost
net_returns = portfolio_returns - cost

# net profit
portfolio_value = (1 + net_returns).cumprod() * initial_capital

log.append(" Step 3: Portfolio Calculated")
log.append(f"Total Return: {(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1):.2%}")
log.append(f"Max Drawdown: {(portfolio_value / portfolio_value.cummax() - 1).min():.2%}")

# basic indicator
years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1
sharpe = net_returns.mean() / net_returns.std() * np.sqrt(12)

log.append(f"CAGR: {cagr:.2%}")
log.append(f"Sharpe Ratio: {sharpe:.2f}")
log.append(f"Avg Monthly Return: {net_returns.mean():.4f}")
log.append(f"Avg Turnover: {turnover.mean():.4f}")
log.append(f"Max Turnover: {turnover.max():.4f}")

# output
log.append("\n Sample Portfolio Value:")
log.append(str(portfolio_value.head(5)))

log.append("\n Sample Returns:")
log.append(str(net_returns.head(5)))

log.append("\n Sample Weights:")
log.append(str(weights.head(2)))

log.append("\n Sample Turnover:")
log.append(str(turnover.head(5)))


with open("debug_report.txt", "w") as f:
    for line in log:
        f.write(line + "\n")

print(" Debug report saved to debug_report.txt")
