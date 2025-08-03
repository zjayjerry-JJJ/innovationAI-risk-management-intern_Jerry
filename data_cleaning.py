

import pandas as pd

sector_prices = pd.read_csv('sector_prices.csv', index_col=0, parse_dates=True)
gdp = pd.read_csv('gdp.csv', index_col=0, parse_dates=True)
cpi = pd.read_csv('cpi.csv', index_col=0, parse_dates=True)
rate = pd.read_csv('rate.csv', index_col=0, parse_dates=True)

# missing value
sector_prices = sector_prices.fillna(method='ffill')
gdp = gdp.fillna(method='ffill')
cpi = cpi.fillna(method='ffill')
rate = rate.fillna(method='ffill')

# data timetype unified
sector_prices.index = pd.to_datetime(sector_prices.index)
gdp.index = pd.to_datetime(gdp.index)
cpi.index = pd.to_datetime(cpi.index)
rate.index = pd.to_datetime(rate.index)

# save cleaned data
sector_prices.to_csv('sector_prices_cleaned.csv')
gdp.to_csv('gdp_cleaned.csv')
cpi.to_csv('cpi_cleaned.csv')
rate.to_csv('rate_cleaned.csv')

print("already cleaned and save cleaned.csv")
