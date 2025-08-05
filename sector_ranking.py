import pandas as pd

# read data
momentum = pd.read_csv('momentum_3m.csv', index_col=0, parse_dates=True)
volatility = pd.read_csv('sector_volatility.csv', index_col=0, parse_dates=True)
regime = pd.read_csv('macro_regime_tags.csv', index_col=0, parse_dates=True)

# change to monthly
momentum_month = momentum.resample('M').last()
volatility_month = volatility.resample('M').last()

# score
momentum_z = (momentum_month - momentum_month.mean()) / momentum_month.std()
volatility_z = (volatility_month - volatility_month.mean()) / volatility_month.std()
score = momentum_z - volatility_z

score.to_csv('sector_monthly_score.csv')
print("monthly score has been saves as sector_monthly_score.csv")

# 3 highest score etf
top_sectors = score.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
top_sectors_df = top_sectors.to_frame(name='Top 3 Sectors')

top_sectors_df.to_csv('monthly_top_sectors.csv')
print("3 top sector has been saved as monthly_top_sectors.csv")

#rebalance
rebalance_schedule = top_sectors_df.copy()
rebalance_schedule['Weights'] = rebalance_schedule['Top 3 Sectors'].apply(lambda lst: [1/3]*3)

rebalance_schedule.to_csv('rebalance_schedule.csv')
print("rebalance has been saved as rebalance_schedule.csv")


# reasonable allocations
weights = score.div(score.sum(axis=1), axis=0) 
weights = weights.clip(upper=0.33)  # limited to 33%
weights = weights.div(weights.sum(axis=1), axis=0)

# drawdown
def max_drawdown(series):
    cum_max=series.cum_max()
    drawdown = (series - cum_max)/cum_max
    return drawdown.min()

#equally weighted
weights = pd.DataFrame(index=score.index, columns=score.columns)
for date in score.index:
    top = score.loc[date].nlargest(3).index
    weights.loc[date,top] = 1/3
weights = weights.fillna(0)
weights.to_csv('weights.csv')

