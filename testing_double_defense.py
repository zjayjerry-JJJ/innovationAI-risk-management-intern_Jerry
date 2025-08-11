import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

initial_capital = 1000000
transaction_cost = 0.001

def apply_dual_defense_strategy(net_returns: pd.Series,
                                 portfolio_value: pd.Series,
                                 floor: float = -0.05,
                                 cost: float = 0.01,
                                 trigger2: float = -0.05) -> pd.Series:
   
    rolling_max = portfolio_value.cummax()
    drawdown = portfolio_value / rolling_max - 1
    in_hedge = drawdown < trigger2

    protected_returns = net_returns.copy()
    hedge_months = set()

    for i, date in enumerate(net_returns.index):
        if in_hedge.iloc[i]:
            ret = net_returns.iloc[i]
            month_key = (date.year, date.month)
            if month_key not in hedge_months:
                ret -= cost
                hedge_months.add(month_key)
            if ret < floor:
                ret = floor - (cost if month_key not in hedge_months else 0)
            protected_returns.iloc[i] = np.clip(ret, -0.999, 1.0)
    return protected_returns

prices = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)
weights = pd.read_csv('weights.csv', index_col=0, parse_dates=True)

weights = weights.reindex(prices.index).ffill().fillna(0)
returns = prices.pct_change().fillna(0)

# some data ratio
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
turnover = (weights - weights.shift(1)).abs().sum(axis=1)
costs = turnover * transaction_cost
net_returns = portfolio_returns - costs
portfolio_value_initial = (1 + net_returns).cumprod() * initial_capital

# first defense line
rolling_max = portfolio_value_initial.cummax()
drawdown = portfolio_value_initial / rolling_max - 1
in_defense = drawdown < -0.035

defense_mode = False
for i in range(len(in_defense)):
    if drawdown.iloc[i] < -0.035:
        defense_mode = True
    elif drawdown.iloc[i] > -0.02:
        defense_mode = False
    in_defense.iloc[i] = defense_mode

# switch to defense etf
defensive_sectors = ['XLU', 'XLV', 'XLP']
defensive_weights = pd.Series(1 / len(defensive_sectors), index=defensive_sectors)

for date in in_defense.index[in_defense]:
    if date in weights.index:
        weights.loc[date] = 0
        for sec in defensive_sectors:
            if sec in weights.columns:
                weights.loc[date, sec] = defensive_weights[sec]

# evaluate pefermance
portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
turnover = (weights - weights.shift(1)).abs().sum(axis=1)
costs = turnover * transaction_cost
net_returns = portfolio_returns - costs
portfolio_value_defensive = (1 + net_returns).cumprod() * initial_capital

#second hedge line
net_returns_hedged = apply_dual_defense_strategy(
    net_returns=net_returns,
    portfolio_value=portfolio_value_defensive,
    floor=-0.05,
    cost=0.01,
    trigger2=-0.05
)

# final value
final_value = (1 + net_returns_hedged).cumprod() * initial_capital

#output
monthly_value = final_value.resample('ME').last().dropna()
monthly_return = monthly_value.pct_change().dropna()
rf = 0.0423 / 12

total_return = final_value.iloc[-1] / final_value.iloc[0] - 1
years = (final_value.index[-1] - final_value.index[0]).days / 365.25
cagr = (final_value.iloc[-1] / final_value.iloc[0]) ** (1 / years) - 1
vol = monthly_return.std() * np.sqrt(12)
rolling_max_m = monthly_value.cummax()
drawdown_m = monthly_value / rolling_max_m - 1
max_drawdown = drawdown_m.min()
excess_return = monthly_return - rf
sharpe = excess_return.mean() / monthly_return.std() * np.sqrt(12)

print("================ Strategy Debug Info ================")
print(f"Total Return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Final Value: {final_value.iloc[-1]:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Annualized Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Min Return (Before): {net_returns.min():.2%}")
print(f"Min Return (After):  {net_returns_hedged.min():.2%}")
print("======================================================")
