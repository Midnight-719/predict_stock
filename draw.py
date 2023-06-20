# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy   as  np
# -----------------------------------------------------#
# # 读取数据
# df = pd.read_csv('500.csv')
#
# # 选择需要生成热力图的数据列
# data = df[['trade_date', 'open', 'close', 'high', 'low', 'vol']]
#
# # 计算相关系数矩阵
# corr = data.corr()
#
# # 绘制热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.1, linecolor='white')
# plt.title('Financial Data Heatmap')
# plt.show()
#----------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('500.csv')

# # 填充缺失值
# df.fillna(method='ffill', inplace=True) # 使用前面的非缺失值填充
# # 插值填充
# df.interpolate(method='linear', inplace=True) # 线性插值

# 折线图
plt.figure(figsize=(10, 6))
plt.plot(df['trade_date'], df['close'], label='Closing Price')
plt.title('Financial Data Line Plot')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()



# 散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['vol'], df['close'], marker='o')
plt.title('Financial Data Scatter Plot')
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.show()

# 柱状图
plt.figure(figsize=(10, 6))
plt.bar(df['trade_date'], df['vol'], label='Volume')
plt.bar(df['trade_date'], df['close'], label='Closing Price')
plt.title('Financial Data Bar Plot')
plt.xlabel('Date')
plt.ylabel('Price/Volume')
plt.legend()
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['open', 'close', 'high', 'low']])
plt.title('Financial Data Box Plot')
plt.xlabel('Price')
plt.ylabel('Variable')
plt.show()

# 热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df[['open', 'close', 'high', 'low', 'vol']].corr(), annot=True, cmap='coolwarm', linewidths=0.1, linecolor='white')
plt.title('Financial Data Heatmap')
plt.show()