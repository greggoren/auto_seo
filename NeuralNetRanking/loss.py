from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class NewHingeLoss(_Loss):


    def __init__(self, margin=1.0, size_average=True):
        super(NewHingeLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average


    def forward(self, input1,input2, target):
        input = input1-input2
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average)