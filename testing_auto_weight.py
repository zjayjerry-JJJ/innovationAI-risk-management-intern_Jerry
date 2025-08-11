import pandas as pd
import numpy as np

# 
TOP_K = 3
LAMBDA_GRID = [0.0, 0.15, 0.3, 0.5]   
TX_COST = 0.001                       
WEIGHTS_OUT = "weights.csv"

# data
prices = pd.read_csv("sector_prices_cleaned.csv", index_col=0, parse_dates=True).sort_index()
rets = prices.pct_change().fillna(0.0)
score_m = pd.read_csv("sector_monthly_score.csv", index_col=0, parse_dates=True).sort_index()
score_m = score_m.resample("ME").last()   # 月末评分

# weighted
ew_m = pd.DataFrame(0.0, index=score_m.index, columns=score_m.columns)
prop_m = pd.DataFrame(0.0, index=score_m.index, columns=score_m.columns)

for dt in score_m.index:
    s = score_m.loc[dt].dropna()
    if s.empty: 
        continue
    topk = s.nlargest(TOP_K)
    sectors = topk.index.tolist()

    # equal weight
    ew_m.loc[dt, sectors] = 1.0 / TOP_K

    # scored weight
    pos = topk.clip(lower=0)
    if pos.sum() <= 0:
        prop_m.loc[dt, sectors] = 1.0 / TOP_K
    else:
        prop_m.loc[dt, sectors] = pos / pos.sum()

# month to daily
def monthly_to_daily(w_m):
    w_d = w_m.reindex(prices.index).ffill().fillna(0.0)
    # match
    return w_d.reindex(columns=prices.columns).fillna(0.0)

ew_d   = monthly_to_daily(ew_m)
prop_d = monthly_to_daily(prop_m)

# testing
def backtest(weights_d):
    port_ret = (weights_d.shift(1) * rets).sum(axis=1).fillna(0.0)
    turnover = (weights_d - weights_d.shift(1)).abs().sum(axis=1).fillna(0.0)
    net_ret = port_ret - TX_COST * turnover
    equity = (1 + net_ret).cumprod()
    tr = equity.iloc[-1] / equity.iloc[0] - 1
    return tr, equity

# best lambda
best_lambda, best_tr, best_w = None, -9e9, None
for lam in LAMBDA_GRID:
    mix_d = (1 - lam) * ew_d + lam * prop_d
    # united
    mix_d = mix_d.div(mix_d.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    tr, _ = backtest(mix_d)
    print(f"λ={lam:.2f}  Total Return={tr:.2%}")
    if tr > best_tr:
        best_tr = tr
        best_lambda = lam
        best_w = mix_d.copy()

print(f"[Chosen] λ={best_lambda:.2f}  Total Return={best_tr:.2%}")

# output
best_w.to_csv(WEIGHTS_OUT)
print(f"Blended weights saved to {WEIGHTS_OUT}")
