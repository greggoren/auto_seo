import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):#TODO: set right input output sizes
    def create_bank1(self, k, input_dim, output_dim):
        for i in range(k):
            tmp = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=64, stride=1)
            self.conv_bank1.append(tmp)

    def __init__(self,in_size,out_size,number_of_filters,feat_dim):
        super(Net, self).__init__()
        # self.conv = nn.conv2d(1,d,m)
        self.pool = nn.MaxPool1d(out_size)
        self.conv_bank1 = nn.ModuleList()
        self.create_bank1(number_of_filters,in_size,out_size)
        self.k = number_of_filters
        self.fc = nn.Linear(self.k*2+feat_dim,1)
        self.rel = nn.ReLU()

    def forward(self,x):
        out = list()
        for i in range(self.k):
            tmp = self.conv_bank1[i](x)
            out.append(self.pool(tmp))
