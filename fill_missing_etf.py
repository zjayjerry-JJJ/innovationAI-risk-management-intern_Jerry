import yfinance as yf
import pandas as pd

prices = pd.read_csv('sector_prices.csv', index_col=0, parse_dates=True)

tickers_to_fill = ['XLC', 'XLRE']

for ticker in tickers_to_fill:
    print(f"Downloading: {ticker}")
    etf = yf.download(ticker, start='2015-01-01')['Close']
    etf.name = ticker
    prices[ticker] = etf

prices.to_csv('sector_prices.csv')
print("missing ETF data has been restored")
