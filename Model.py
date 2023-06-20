import torch.nn as nn


class lstm(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers , output_size, dropout, batch_first=True):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=self.batch_first, dropout=self.dropout )
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        :param x: (batch_size=100,seq_len=150,features=5)
         out: (100,150,64)  out_t 是当前时间步的输出信息 out 是所有时间步的输出信息 例如可以通过 out_t = out[:, -1, :] 来获取最后一个时间步的 out 的输出张量，维度是 [batch_size, hidden_size]
         hidden: (1,100,64) 就是LSTM隐藏层最后一个时间步输出的
         cell: (1,100,64)
        :return:
        """
        out, (hidden, cell) = self.lstm(x)
        out = self.linear(hidden)
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers , output_size, dropout, batch_first=True):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out,hidden = self.rnn(x)

        out = self.fc(hidden)
        return out

import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers , output_size, dropout, batch_first=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial hidden state
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #
        # # Forward propagate GRU
        # out, _ = self.gru(x, h0)
        #
        # # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        out, hidden = self.gru(x)

        out = self.fc(hidden)
        return out

#残差LSTM
# 定义带有残差连接的三层LSTM模型
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers , output_size, dropout, batch_first=True):
        super(ResidualLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.lstm1= nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=self.batch_first, dropout=self.dropout )
        self.lstm2= nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=self.batch_first, dropout=self.dropout )
        self.lstm3= nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=self.batch_first, dropout=self.dropout )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out_1, _ = self.lstm1(x)
        out_2, _ = self.lstm2(out_1)
        out_3, _ = self.lstm3(out_2)
        residual = out_1 + out_3
        out = self.fc(residual[:,-1,:])
        return out


class ResidualLSTM5(nn.Module):
    def __init__(self,  input_size, hidden_size, num_layers , output_size, dropout, batch_first=True):
        super(ResidualLSTM5, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.lstm1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,batch_first=self.batch_first, dropout=self.dropout)
        self.lstm2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,batch_first=self.batch_first, dropout=self.dropout)
        self.lstm3 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,batch_first=self.batch_first, dropout=self.dropout)
        self.lstm4 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,batch_first=self.batch_first, dropout=self.dropout)
        self.lstm5 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,batch_first=self.batch_first, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out_1, _ = self.lstm1(x)
        out_2, _ = self.lstm2(out_1)
        out_3, _ = self.lstm3(out_2)
        out_4, _ = self.lstm4(out_3)
        out_5, _ = self.lstm5(out_4)
        residual = out_1 + out_2 + out_3 + out_4 + out_5
        out = self.fc(residual[:,-1,:])
        return out