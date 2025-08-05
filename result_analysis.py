# 宏观变量与行业表现的可视化分析
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据（根据你的实际路径调整）
gdp = pd.read_csv('gdp.csv', index_col=0, parse_dates=True)
cpi = pd.read_csv('cpi.csv', index_col=0, parse_dates=True)
rate = pd.read_csv('rate.csv', index_col=0, parse_dates=True)
sector_prices = pd.read_csv('sector_prices_cleaned.csv', index_col=0, parse_dates=True)

# 月度重采样
monthly_sector = sector_prices.resample('M').last()
monthly_returns = monthly_sector.pct_change().dropna()

# 合并宏观变量（使用 last 方式对齐）
macro = pd.concat([
    gdp.resample('M').last().rename(columns={gdp.columns[0]: 'GDP'}),
    cpi.resample('M').last().rename(columns={cpi.columns[0]: 'CPI'}),
    rate.resample('M').last().rename(columns={rate.columns[0]: 'Rate'})
], axis=1).dropna()

# 合并收益数据与宏观变量
data = pd.concat([monthly_returns, macro], axis=1).dropna()

# 画图 - 宏观变量与不同行业的相关性热力图
corr_matrix = data.corr().loc[['GDP', 'CPI', 'Rate'], monthly_returns.columns]
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Macro Variables and Sector Returns')
plt.tight_layout()
plt.savefig('macro_sector_correlation.png')
plt.show()

# 画图 - 宏观变量变化与行业平均收益
macro_conditions = pd.cut(data['GDP'], bins=3, labels=['Low GDP', 'Medium GDP', 'High GDP'])
mean_returns_by_gdp = data.groupby(macro_conditions)[monthly_returns.columns].mean()

mean_returns_by_gdp.T.plot(kind='bar', figsize=(12,6))
plt.title('Average Sector Returns under Different GDP Growth Levels')
plt.ylabel('Average Monthly Return')
plt.grid(True)
plt.tight_layout()
plt.savefig('gdp_vs_sector_returns.png')
plt.show()
