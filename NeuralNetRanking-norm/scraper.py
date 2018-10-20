import torch
a = torch.DoubleTensor([1]).cuda()
print(a)
b = torch.DoubleTensor([-1]).cuda()
print(a+b)