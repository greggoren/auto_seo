import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleRankNet(nn.Module):
    def __init__(self,vec_dimension,out2,out3,dropout_p):
        super(SimpleRankNet,self).__init__()
        self.layer1 = nn.Linear(vec_dimension*3,vec_dimension*3)
        self.layer2 = nn.Linear(vec_dimension*3,out2)
        self.layer3 = nn.Linear(out2,out3)
        self.layer4 = nn.Linear(vec_dimension * 3, vec_dimension * 3)
        self.layer5 = nn.Linear(vec_dimension * 3, out2)
        self.layer6 = nn.Linear(out2, out3)
        self.vec_dimension = vec_dimension
        self.droput = nn.Dropout(dropout_p)
        self.norm1 = nn.BatchNorm1d(vec_dimension*3)
        self.norm2 = nn.BatchNorm1d(out2)


    def define_block1(self,x):
        x = F.relu(self.norm1(self.layer1(x)))
        x = self.droput(x)
        x = F.relu(self.norm2(self.layer2(x)))
        x = self.droput(x)
        x = self.layer3(x)
        return x


    def define_block2(self,x):
        x = F.relu(self.norm1(self.layer4(x)))
        x = self.droput(x)
        x = F.relu(self.norm2(self.layer5(x)))
        x = self.droput(x)
        x = self.layer6(x)
        return x



    # def split_data(self,x):
    #     sentence_comb1 = torch.cat((x[0],x[1],x[2]),1)
    #     sentence_comb2 = torch.cat((x[2],x[3],x[4]),1)
    #     return sentence_comb1,sentence_comb2

    def forward(self,x):
        comb1,comb2 = x[0],x[1]
        out1 = self.define_block1(comb1)
        out2 = self.define_block2(comb2)
        return out1,out2


