import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

initial_capital = 1000000
transaction_cost = 0.001

# --- Protected Put 对冲机制函数（修正版本） ---
def apply_option_hedge_only(net_returns: pd.Series,
                             portfolio_value: pd.Series,
                             floor: float = -0.05,
                             cost: float = 0.01,
                             trigger_drawdown: float = -0.035) -> pd.Series:
    rolling_max = portfolio_value.cummax()
    drawdown = portfolio_value / rolling_max - 1
    in_hedge = drawdown < trigger_drawdown

    protected_returns = net_returns.copy()
    hedge_flag = False

    for i in range(len(net_returns)):
        date = net_returns.index[i]
        if in_hedge.iloc[i]:
            if not hedge_flag:
                # 第一次进入对冲状态，扣除 cost
                hedge_flag = True
                effective_cost = cost
            else:
                effective_cost = 0  # 后续不重复扣除成本

            raw_ret = net_returns.iloc[i]
            if raw_ret < floor:
                protected = floor
            else:
                protected = raw_ret
            protected -= effective_cost
            protected = np.clip(protected, -0.999, 1.0)
            protected_returns.iloc[i] = protected
        else:
            hedge_flag = False  # 退出对冲状态

    return protected_returns


# --- 数据读取 ---
prices = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)
weights = pd.read_csv('weights.csv', index_col=0, parse_dates=True)

# 对齐时间序列
weights = weights.reindex(prices.index).ffill().fillna(0)
returns = prices.pct_change().fillna(0)

# 初始净值计算（不做任何防御）
portfolio_returns_initial = (weights.shift(1) * returns).sum(axis=1)
turnover_initial = (weights - weights.shift(1)).abs().sum(axis=1)
cost_initial = turnover_initial * transaction_cost
net_returns_initial = portfolio_returns_initial - cost_initial
portfolio_value_initial = (1 + net_returns_initial).cumprod() * initial_capital

# 应用 Protected Put 对冲（单层防御）
net_returns_protected = apply_option_hedge_only(
    net_returns=net_returns_initial,
    portfolio_value=portfolio_value_initial,
    floor=-0.05,
    cost=0.01,
    trigger_drawdown=-0.035
)

# 最终净值计算
net_returns_protected = net_returns_protected.fillna(0).replace([np.inf, -np.inf], 0)

portfolio_value = (1 + net_returns_protected).cumprod()
portfolio_value *= initial_capital
portfolio_value = (1 + net_returns_protected).cumprod() * initial_capital
portfolio_value.to_csv('portfolio_value.csv')

# 指标计算
monthly_value = portfolio_value.resample('ME').last().dropna()
monthly_return = monthly_value.pct_change().dropna()
rf = 0.0423 / 12

total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1
years = (portfolio_value.index[-1] - portfolio_value.index[0]).days / 365.25
cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1
vol = monthly_return.std() * np.sqrt(12)
rolling_max_m = monthly_value.cummax()
drawdown_m = monthly_value / rolling_max_m - 1
max_drawdown = drawdown_m.min()
excess_return = monthly_return - rf
sharpe = excess_return.mean() / monthly_return.std() * np.sqrt(12)

# 打印关键指标
print("================ Portfolio Performance ================")
print(f"Total Return: {total_return:.2%}")
print(f"CAGR: {cagr:.2%}")
print(f"Annualized Volatility: {vol:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print("=======================================================")
print("Initial Portfolio Value Head:")
print(portfolio_value_initial.head())
print("Initial Portfolio Value Tail:")
print(portfolio_value_initial.tail())
print("Any NaN in initial value?", portfolio_value_initial.isna().sum())

print("\nInitial Net Returns (head):")
print(net_returns_initial.head())
print("Min/Max Initial Returns:", net_returns_initial.min(), net_returns_initial.max())

print("\nProtected Net Returns (head):")
print(net_returns_protected.head())
print("Min/Max Protected Returns:", net_returns_protected.min(), net_returns_protected.max())
print("Final Portfolio Value:", portfolio_value.iloc[-1])
