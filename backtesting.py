import pandas as pd
import numpy as np
import seaborn as sns


initial_capital = 1000000
transaction_cost = 0.001


# read data
prices = pd.read_csv('sector_prices_cleaned.csv',index_col=0,parse_dates=True)
weights = pd.read_csv('weights.csv', index_col=0, parse_dates=True)

# match
weights = weights.reindex(prices.index).ffill().fillna(0)

# returns
returns = prices.pct_change().fillna(0)

# portfolio_returns
portfolio_returns = (weights.shift(1) * returns).sum(axis = 1)

# max drawdown adjusted to defense mode
portfolio_returns_initial = (weights.shift(1) * returns).sum(axis=1)
turnover_initial = (weights - weights.shift(1)).abs().sum(axis=1)
cost_initial = turnover_initial * transaction_cost
net_returns_initial = portfolio_returns_initial - cost_initial
portfolio_value_initial = (1 + net_returns_initial).cumprod() * initial_capital

rolling_max = portfolio_value_initial.cummax()
drawdown = portfolio_value_initial / rolling_max - 1

in_defense = drawdown < -0.038
defense_flag = False
for i in range(len(drawdown)):
    if in_defense.iloc[i]:
        defense_flag = True
    elif portfolio_value_initial.iloc[i] >= rolling_max.iloc[i]:  
        defense_flag = False
    in_defense.iloc[i] = defense_flag

# apply defense
defensive_sectors = ['XLU', 'XLV', 'XLP']
for dt in in_defense[in_defense].index:
    if dt in weights.index:
        weights.loc[dt] = weights.shift(1).loc[dt]

# turnover
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
turnover = (weights - weights.shift(1)).abs().sum(axis=1)
cost = turnover * transaction_cost
net_returns = portfolio_returns - cost

# net profit
portfolio_value = (1 + net_returns).cumprod() * initial_capital



portfolio_value.to_csv('portfolio_value.csv')

# cumulative profit
total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1

#time 
years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
cagr = (portfolio_value.iloc[-1] / portfolio_value[0]) ** (1/years) - 1

# annualize V
vol = net_returns.std() * np.sqrt(12)

# sharp ratio
monthly_value = portfolio_value.resample('M').last().dropna()
monthly_return = monthly_value.pct_change().dropna()
rf = 0.0423 / 12

# max drawdown
rolling_max_m = monthly_value.cummax()
drawdown_m = monthly_value / rolling_max_m - 1
max_drawdown = drawdown_m.min()

excess_return = monthly_return - rf
sharpe = excess_return.mean() / monthly_return.std() * np.sqrt(12)

#annualize V
monthly_value = portfolio_value.resample('M').last().dropna()
monthly_return = monthly_value.pct_change().dropna()
annualized_vol = monthly_return.std() * np.sqrt(12)

print(f"Total Return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Annualized Volatility: {annualized_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

import matplotlib.pyplot as plt

# read data
portfolio = pd.read_csv('portfolio_value.csv', index_col=0, parse_dates=True)
spy = pd.read_csv('spy_monthly.csv', index_col=0, parse_dates=True)
equal_weight = pd.read_csv('equal_weight_value.csv', index_col=0, parse_dates=True)

# unify
portfolio.columns = ['Strategy']
spy.columns = ['SPY']
equal_weight.columns = ['EqualWeight']

# time matcha
portfolio = portfolio.resample('ME').last()
spy = spy.resample('ME').last()
equal_weight = equal_weight.resample('ME').last()

# combine
combined = pd.concat([portfolio, spy, equal_weight], axis=1)
combined = combined.dropna()

normalized = combined / combined.iloc[0]

# picture
plt.figure(figsize=(12, 6))
for col in normalized.columns:
    plt.plot(normalized[col], label=col)
plt.title('My Strategy vs SPY vs Equal-Weighted Benchmark')
plt.ylabel('Normalized Portfolio Value')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('strategy_vs_all_benchmarks.png')
plt.show()

# picture maximum drawdowns
drawdown = combined['Strategy'] / combined['Strategy'].cummax() - 1
plt.figure(figsize=(12,4))
plt.plot(drawdown, label = 'Strategy Drawdown', color = 'red')
plt.title('Strategy Drawdown Over time')
plt.ylabel('Drawdown')
plt.xlabel('Date')
plt.grid(True)
plt.tight_layout()
plt.savefig('strategy_drawdown.png')
plt.show()

# monthly allocation
monthly_weights = weights.resample('ME').last().dropna(how='all')
top_sectors = pd.DataFrame(columns=['Top1', 'Top2', 'Top3'], index=monthly_weights.index)

for date, row in monthly_weights.iterrows():
    top = row.nlargest(3).index
    top_sectors.loc[date] = top

one_hot = pd.DataFrame(0, index=top_sectors.index, columns=monthly_weights.columns)

for date in top_sectors.index:
    for sector in top_sectors.loc[date]:
        one_hot.loc[date, sector] = 1
plt.figure(figsize=(12, 6))
sns.heatmap(one_hot.T, cmap='Greens', cbar=False, linewidths=0.1, linecolor='gray')
plt.title("Monthly Top 3 Sector Allocation (Heatmap)")
plt.ylabel("Sectors")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("monthly_allocation_heatmap.png")
plt.show()

# sector rotation heatmap


rolling_sharpe = net_returns.rolling(12).mean() / net_returns.rolling(12).std() * np.sqrt(12)

plt.figure(figsize=(10, 4))
rolling_sharpe.plot()
plt.title("Rolling 12-Month Sharpe Ratio")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.tight_layout()
plt.savefig("rolling_sharpe.png")
plt.show()