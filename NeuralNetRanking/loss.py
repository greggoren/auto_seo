from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class NewHingeLoss(_Loss):


    def __init__(self, margin=1.0, size_average=True):
        super(NewHingeLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average


    def forward(self, input1,input2, target):
        return F.margin_ranking_loss(input1, input2, self.margin, self.size_average)