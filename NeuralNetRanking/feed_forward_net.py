import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleRankNet(nn.Module):
    def __init__(self,vec_dimension,out2,out3):
        super(SimpleRankNet,self).__init__()
        self.layer1 = nn.Linear(vec_dimension*3,vec_dimension*3)
        self.layer2 = nn.Linear(vec_dimension*3,out2)
        self.layer3 = nn.Linear(out2,out3)
        self.vec_dimension = vec_dimension


    def define_block(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def split_data(self,x):
        # query_index = self.vec_dimension*2+1
        sentence_comb1 = torch.cat((x[0],x[1],x[2]),0)
        sentence_comb2 = torch.cat((x[2],x[3],x[4]),0)
        return sentence_comb1,sentence_comb2

    def forward(self,x):
        comb1,comb2 = self.split_data(x)
        out1 = self.define_block(comb1)
        out2 = self.define_block(comb2)
        # last_input_for_layer = torch.cat((out1,out2),0)
        # final_out = self.layer3(last_input_for_layer)
        return out1,out2


