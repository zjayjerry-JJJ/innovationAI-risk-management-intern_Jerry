import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#read cleaned csv
df = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)

# MACD 30
mavg = df.rolling(window=30).mean()
mavg.to_csv('sector_mavg.csv')
print("MACD 30 saved in sector_mavg.csv")

# 30 days return rate
returns = df.pct_change(periods=30)
returns.to_csv('sector_return.csv')
print("30 days return rate saved in sector_return.csv")

# 30 days annualized Volatility
volatility = df.pct_change().rolling(window=30).std() * np.sqrt(252)
volatility.to_csv('sector_volatility.csv')
print("30 days annualized Volatility saved in sector_volatility.csv")


# unify
prices = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)
mavg = pd.read_csv('sector_mavg.csv', index_col=0, parse_dates=True)
volatility = pd.read_csv('sector_volatility.csv', index_col=0, parse_dates=True)
gdp = pd.read_csv('gdp_cleaned.csv', index_col=0, parse_dates=True)
cpi = pd.read_csv('cpi_cleaned.csv', index_col=0, parse_dates=True)
rate = pd.read_csv('rate_cleaned.csv', index_col=0, parse_dates=True)

# etf price trend
plt.figure(figsize=(12, 6))
prices.plot(ax=plt.gca())
plt.title('Sector ETF Closing Prices')
plt.ylabel('Price')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('etf_prices.png')
plt.close()

# moving average vs price trend
sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLU', 'XLB', 'XLRE', 'XLP', 'XLC']

for ticker in sectors:
    plt.figure(figsize=(12, 6))
    plt.plot(prices[ticker], label=f'{ticker} Price')
    plt.plot(mavg[ticker], label=f'{ticker} 30-day MA')
    plt.legend()
    plt.title(f'{ticker}: Price vs 30-day Moving Average')
    plt.tight_layout()
    plt.savefig(f'{ticker.lower()}_mavg.png')  
    plt.close()

# volatility trend
plt.figure(figsize=(12, 6))
plt.plot(volatility['XLF'], label='XLF Volatility')
plt.plot(volatility['XLE'], label='XLE Volatility')
plt.plot(volatility['XLV'], label='XLV Volatility')
plt.legend()
plt.title('Sector Rolling Volatility (30-day)')
plt.tight_layout()
plt.savefig('sector_volatility.png')
plt.close()

# macro trend(GDP)
plt.figure(figsize=(12, 6))
plt.plot(gdp, label='GDP')
plt.title('GDP over Time')
plt.tight_layout()
plt.savefig('gdp_trend.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(cpi, label='CPI')
plt.title('CPI over Time')
plt.tight_layout()
plt.savefig('cpi_trend.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(rate, label='Federal Funds Rate')
plt.title('Federal Rate over Time')
plt.tight_layout()
plt.savefig('rate_trend.png')
plt.close()

print("picture already saved")
