



from Model import lstm,RNN,GRU
from data_loader import getdata
import  torch
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

input_size = 5
hidden_size = 64
num_layers = 2
output_size = 10
learning_rate = 0.001
epochs = 100
batch_size = 100
seq_len = 15
batch_first = True
# shuffle = True
dropout =0
num_workers =4
train_sampler   = None
val_sampler     = None
# model = GRU(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
model = lstm(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
# model = RNN(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load('net_paramslstm-150-2-50.pth')
# checkpoint = torch.load('net_params150-50.pth')
model.load_state_dict(checkpoint['state_dict'])
preds = []
labels = []
file = "SP500_401_520.csv"
sp500 = pd.read_csv(file)
data = sp500[['open', 'close', 'high', 'low', 'vol']]
data.fillna(method='ffill', inplace=True)
data = np.array(data)

file = "500.csv"

train_dataset_regression, test_dataset_regression, train_loader, test_loader,min_val,max_val = getdata(file,seq_len,batch_size)
model.eval()

# val--> data
val = torch.tensor(data, dtype=torch.float32).cuda()
pred = model(val)

list= pred.data.squeeze(1).tolist()
preds.extend(list[-1]) # torch.Size([1, 32, 1])
# labels.extend(label.tolist()) # torch.Size([32, 1])
# 反归一化
original_preds   = []
original_labels  = []
YYY = []
for l in range(0,len(preds)):
    ll = preds[l][0]
    lll = (max_val[1] - min_val[1])*ll + min_val[1]
    original_preds.append(lll)

for j in range(0,len(labels)):
    jj = labels[j][0]
    jjj = (max_val[1] - min_val[1])*jj + min_val[1]
    original_labels.append(jjj)
# 验证 反归一化
for k in range(0,len(test_dataset_regression)):
    kk = test_dataset_regression[k][0]
    kkk =  (max_val[1] - min_val[1])*kk + min_val[1]
    YYY.append(kkk)

plt.plot(original_preds, label='Predicted')
plt.plot(original_labels, label='original')
#plt.plot(YYY, label='yyy')
plt.legend()
plt.show()

# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(original_preds , original_labels)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(original_preds , original_labels))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(original_preds , original_labels)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)






