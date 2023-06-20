
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from torchvision import transforms
class stockDataset(Dataset):
    '''
    传入进入的X：train_dataset Y:train_dataset_regression or test
    self.length = len(train_dataset)
    '''
    def __init__(self,X,Y,transform=None):
        super(stockDataset,self).__init__()
        self.x = X
        self.y = Y
        self.length = len(X)
        self.tranform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_n = self.x[index]
        y_n = self.y[index]
        if self.tranform != None:
            return self.tranform(x_n), y_n

        return x_n, y_n  #data, label

# 多天标签预测多天
def getdata(stock_csv,sequence_length,batchSize,time_pridict):
    sp500           = pd.read_csv(stock_csv)
    data            = sp500[['open', 'close', 'high', 'low', 'vol']]
    data.fillna(method='ffill', inplace=True)
    #data.interpolate(method='linear', inplace=True)
    #划分训练集和测试集
    train_num       =  int(0.8*data.shape[0])
    test_num        = data.shape[0]-train_num
    train           = data.iloc[:train_num, :]
    test            = data.iloc[train_num:, :]
    train_close     = train.iloc[:train_num, 1:2].values
    test_close      = test.iloc[:test_num, 1:2].values
    # 归一化==>训练集使用fit_transform,测试集使用transform 记录归一化的min和max
    #min_val = data.min()
    #max_val = data.max()
    sc              = MinMaxScaler(feature_range=(0, 1))
    sc.fit(data)
    min_val         = sc.data_min_
    max_val         = sc.data_max_
    train           = sc.fit_transform(train)
    test            = sc.transform(test)
    train_close     = sc.fit_transform(train_close)
    test_close      = sc.transform(test_close)


    #15个步长预测下一个收盘价(时间步长就是sequence_length )
    time_step       =  sequence_length
    time_pridict    =  time_pridict
    #(1931, 150, 5)
    train_dataset   = np.array([train[i : i + time_step, :] for i in range(0,train.shape[0] - (time_step+time_pridict),1)])
    #(363, 150, 5)
    test_dataset    = np.array([test[i : i + time_step, :] for i in range(0,test.shape[0] - (time_step+time_pridict),1)])

    # train_dataset_regression    = np.array([train_close [i + time_step+  time_pridict ] for i in range(0,train.shape[0] - (time_step+time_pridict),1)])
    train_close = train_close[time_step-1:]
    train_dataset_regression = []
    for i in range(0, train.shape[0] - (time_step + time_pridict), 1):
        train_dataset_regression.append(train_close[i : i+time_pridict])
    train_dataset_regression = np.array(train_dataset_regression)
    train_dataset_regression= np.squeeze(train_dataset_regression, axis=2)

    # test_dataset_regression    = np.array([test_close [i + time_step +  time_pridict ] for i in range(0,test.shape[0] - (time_step+time_pridict),1)])
    test_close = test_close[time_step - 1:]
    test_dataset_regression = []
    for i in range(0, test.shape[0] - (time_step + time_pridict), 1):
        test_dataset_regression.append(test_close[i: i + time_pridict])
    test_dataset_regression = np.array(test_dataset_regression)
    test_dataset_regression= np.squeeze(test_dataset_regression, axis=2)
    """
    train_loader--> DataLoader len(train_loader)=51个batch, 51*32=1632 index:51个（0--50）
    """
    train_loader = DataLoader(dataset=stockDataset(train_dataset ,train_dataset_regression , transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=stockDataset(test_dataset, test_dataset_regression,transform=transforms.ToTensor()), batch_size=batchSize, shuffle=False)

    return train_dataset_regression, test_dataset_regression, train_loader, test_loader,min_val,max_val

# 使用一天的模型预测多天
def get_data(stock_csv,sequence_length,batchSize):
    sp500           = pd.read_csv(stock_csv)
    data            = sp500[['open', 'close', 'high', 'low', 'vol']]
    data.fillna(method='ffill', inplace=True)
    #data.interpolate(method='linear', inplace=True)
    #划分训练集和测试集
    train_num       =  int(0.8*data.shape[0])
    test_num        = data.shape[0]-train_num
    train           = data.iloc[:train_num, :]
    test            = data.iloc[train_num:, :]
    train_close     = train.iloc[:train_num, 1:2].values
    test_close      = test.iloc[:test_num, 1:2].values
    # 归一化==>训练集使用fit_transform,测试集使用transform 记录归一化的min和max
    #min_val = data.min()
    #max_val = data.max()
    sc              = MinMaxScaler(feature_range=(0, 1))
    sc.fit(data)
    min_val         = sc.data_min_
    max_val         = sc.data_max_
    train           = sc.fit_transform(train)
    test            = sc.transform(test)
    train_close     = sc.fit_transform(train_close)
    test_close      = sc.transform(test_close)

    # 15个步长预测下一个收盘价(时间步长就是sequence_length )
    time_step = sequence_length
    time_pridict = 1

    train_dataset = list([train[i: i + time_step, :] for i in range(0, train.shape[0] - (time_step + time_pridict), 1)])
    test_dataset = list([test[i: i + time_step, :] for i in range(0, test.shape[0] - (time_step + time_pridict), 1)])
    # test_dataset_regression    = np.array([test_close [i + time_step +  time_pridict ] for i in range(0,test.shape[0] - (time_step+time_pridict),1)])
    train_dataset_regression = list(
        [train_close[i + time_step + time_pridict] for i in range(0, train.shape[0] - (time_step + time_pridict), 1)])
    test_dataset_regression = list(
        [test_close[i + time_step + time_pridict] for i in range(0, test.shape[0] - (time_step + time_pridict), 1)])
    """
    train_loader--> DataLoader len(train_loader)=51个batch, 51*32=1632 index:51个（0--50）
    """
    train_loader = DataLoader(
        dataset=stockDataset(train_dataset, train_dataset_regression, transform=transforms.ToTensor()),
        batch_size=batchSize,
        shuffle=True)
    test_loader = DataLoader(dataset=stockDataset(test_dataset, test_dataset_regression), batch_size=batchSize,
                             shuffle=False)

    return train_dataset_regression, test_dataset_regression, train_loader, test_loader, min_val, max_val



