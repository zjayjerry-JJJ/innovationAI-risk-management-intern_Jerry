from fredapi import Fred
import yfinance as yf
import pandas as pd
import shutil
import sqlite3
etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLU', 'XLB','XLRE','XLP','XLC']
prices = yf.download(etfs, 
                     start="2015-01-01", 
                     end="2025-01-01", 
                     progress=False, 
                     auto_adjust=True,
                     threads=False)


# 1. saved data
adj_close = prices['Close'] 

adj_close.to_csv('sector_prices.csv')
print("Historical close price has been saved in sector_prices.csv")

fred = Fred(api_key='80ed660e9d2d6c9c9583c4783a0d2240')


#2. Macro data
#GDP
gdp = fred.get_series('GDP').resample('Q-DEC').last()
gdp.to_csv('gdp.csv')
print("GDP data has been saved in gdp.csv")
#CPI
cpi = fred.get_series('CPIAUCSL').resample('ME').last()
cpi.to_csv('cpi.csv')
print("CPI data has been saved in cpi.csv")
#Federal rate
rate = fred.get_series('FEDFUNDS').resample('ME').last()
rate.to_csv('rate.csv')
print("Federal rate has been save in rate.csv")
#three-years treasury bill
t3m = fred.get_series('DGS3MO').resample('ME').last()
t3m.to_csv('treasury_3m.csv')
#ten-years treasury note
t10y = fred.get_series('DGS10').resample('ME').last()
t10y.to_csv('treasury_10y.csv')


#3.Financial data ratio 
import pandas as pd
df = pd.read_csv('sector_pe_pb.csv', index_col=0)
print(df)
pe_pb = pd.read_csv('sector_pe_pb.csv', index_col=0)

conn = sqlite3.connect('market_data.db')


# 4.Saved all the file in SQLite
datasets = {
    'sector_prices': 'sector_prices.csv',
    'gdp': 'gdp.csv',
    'cpi': 'cpi.csv',
    'rate': 'rate.csv',
    'sector_pe_pb': 'sector_pe_pb.csv'
}

for table, file in datasets.items():
    df = pd.read_csv(file)
    df.to_sql(table, conn, if_exists='replace', index=False)
    print(f" wrote {file} in dataset {table}")

#missing data
prices = pd.read_csv('sector_prices.csv')
prices.to_sql('sector_prices', conn, if_exists='replace', index=False)

conn.close()



