import  pandas as pd
import matplotlib.pyplot as plt

"""
不要运行  这个会反向
"""
# # 导入tushare
# import tushare as ts
# # 初始化pro接口
# pro = ts.pro_api('547c4af47532d8cfdf2729518bdf9b7f617c30bba5b09b723916e9c4')
#
# # 拉取数据
# df = pro.index_global(**{
#     "ts_code": "SPX",
#     "trade_date": "",
#     "start_date": 2013,
#     "end_date": 20230530,
#     "limit": "",
#     "offset": ""
# }, fields=[
#     "ts_code",
#     "trade_date",
#     "open",
#     "close", # 收盘价
#     "high",
#     "low",
#     "pre_close",
#     "vol"
# ])
"""
#把在tushare上拉去到的数据写成csv的格式
df类型是pandas.core.frame.DataFrame
我们首先使用 Tushare 下载了某股票的历史行情数据，并将其保存在 pandas 的 DataFrame 中。
接着，使用 DataFrame 的 to_csv 方法将数据保存为 csv 文件。
其中，encoding 参数指定了文件编码格式，utf-8-sig 可以确保在 Excel 中正确显示中文。
将get_data代码中的股票代码替换成你需要的金融数据，然后运行代码即可将数据转换为 csv 文件。
"""
# df.to_csv("pp500.csv",encoding ="utf-8-sig")



# plt.figure(figsize=(16, 8)) #设定图的长和宽
# plt.title('s&p500')
# plt.plot(df['close'])
# plt.show()




stock_csv       = 'pp500.csv'
sp500           = pd.read_csv(stock_csv)


# data            = sp500.iloc[::-1].reset_index(
#
#
# drop=True)
data = sp500[2500:2620]
# data = sp500
# data.to_csv(stock_csv, index=False)

plt.figure(figsize=(16,8)) # 自定义窗口大小

plt.plot(data['close'])


#plt.plot(sp500['close'])



plt.show()

"""
10 fearture
"""


