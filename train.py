import torch.optim.adam
from data_loader import getdata, get_data
from Model import lstm,RNN,GRU,ResidualLSTM,ResidualLSTM5
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import  datetime
from torch.utils.data import DataLoader
#定义参数
input_size = 5
hidden_size = 64
num_layers = 1
output_size = 1
time_pridict=1
learning_rate = 0.001
epochs = 10
batch_size = 100
seq_len = 150
batch_first = True
shuffle = False
dropout =0.1
num_workers =4
train_sampler   = None
val_sampler     = None



# model = GRU(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
# model = lstm(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
# model = RNN(input_size, hidden_size, num_layers, output_size,dropout, batch_first)
model = ResidualLSTM5(input_size, hidden_size, num_layers, output_size,dropout, batch_first)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#file = "s&p500.csv"
file = "500.csv"
train_dataset_regression, test_dataset_regression, train_loader, test_loader ,min_val, max_val= getdata(file,seq_len,batch_size,time_pridict=1)
train_loss = []
for i in range(epochs): # epochs = 50
    model.train()
    running_loss = 0
    for index, (data,label)  in enumerate(train_loader):

        data1 = data.squeeze(1).cuda()
        data1 = torch.tensor(data1 , dtype=torch.float32)
        pred = model(data1.cuda())
        # pred = pred[0, :, :] # (bs,output_size)
        label = torch.tensor(label, dtype=torch.float32).cuda()
        loss = criterion(pred, label)
        optimizer.zero_grad() # 去掉最后一轮的梯度
        loss.backward() # 计算网络学习的参数梯度
        optimizer.step() # 更新模型
        running_loss += loss.item()
        train_loss.append(running_loss/len(train_loader)) # running_loss/len(train_loader 得到每批次的平均损失
    # print(running_loss)
    # if i % 10 == 0:
    #     # torch.save(my_model, args.save_file)
    #     torch.save(model.state_dict(), 'net_params.pth')  # 只保存模型参数
    #     print('第%d epoch，保存模型' % i)
    if epochs==1 or epochs % 10 ==0:
        print("{} Epoch {}, Training loss {}".format(datetime.datetime.now(),i+1,running_loss/len(train_loader)))
        torch.save(model.state_dict(), 'net_params.pth') # 模块没有结构只有权重
    # torch.save(my_model,args.save_file)
# 在使用这种保存模型的时候要先定义模型，再加载模型
torch.save({'state_dict': model.state_dict()},'LSTM-200-1L-50E-10DAY.pth' )
# torch.save(model.state_dict(),'LSTM-200-1L-50E-10DAY.pth')

with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_loss))
# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


train_loss_path = r"./train_loss.txt"  # 存储文件路径

y_train_loss = data_read(train_loss_path)  # loss值，即y轴
x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

plt.figure()

# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')  # x轴标签
plt.ylabel('loss')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
plt.legend()
plt.title('Loss curve')
plt.show()
