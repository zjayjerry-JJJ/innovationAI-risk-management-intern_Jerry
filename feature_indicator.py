import pandas as pd
import numpy as np

# read data
df = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)

# MACD 50 and 200
ma_50 = df.rolling(window=50).mean()
ma_200 = df.rolling(window=200).mean()
ma_50.to_csv('ma_50.csv')
ma_200.to_csv('ma_200.csv')

# momentum
momentum_3m = df.pct_change(periods=63)
momentum_3m.to_csv('momentum_3m.csv')

# rsi
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi_df = df.apply(compute_rsi)
rsi_df.to_csv('rsi_14.csv')

print("indicator has been saved as ma_50.csv, ma_200.csv, momentum_3m.csv, rsi_14.csv")

#Macro expension or contraction

gdp = pd.read_csv('gdp_cleaned.csv', index_col=0, parse_dates=True)
gdp['gdp'] = pd.to_numeric(gdp['gdp'], errors='coerce')


# tag
gdp['YoY_Growth'] = gdp['gdp'].pct_change(periods=4)
gdp['Regime'] = gdp['YoY_Growth'].apply(lambda x: 'Expansion' if x > 0 else 'Contraction')

# saved
gdp[['Regime']].to_csv('macro_regime_tags.csv')
print("Macro regime has been saved as macro_regime_tags.csv")

# picking strong/weak sectors
momentum = pd.read_csv('momentum_3m.csv', index_col=0, parse_dates=True)
volatility = pd.read_csv('sector_volatility.csv', index_col=0, parse_dates=True)

# z-score
momentum_z = (momentum - momentum.mean()) / momentum.std()
volatility_z = (volatility - volatility.mean()) / volatility.std()

# high momentum and low V is better
ranking_score = momentum_z - volatility_z

# saved result
ranking_score.to_csv('sector_ranking_signal.csv')
print("sector has been saved as sector_ranking_signal.csv")