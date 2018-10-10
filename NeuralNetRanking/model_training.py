import torch.optim as optim
import torch.nn as nn
from NeuralNetRanking.feed_forward_net import SimpleRankNet
from NeuralNetRanking.pairwise_data import PairWiseDataLoaer
from torch.utils.data import DataLoader
# class NewHingeLoss(_Loss):

    # def __init__(self, margin=1.0, size_average=True):
    #     super(NewHingeLoss, self).__init__()
    #     self.margin = margin
    #     self.size_average = size_average
    #
    #
    # def forward(self, input, target):
    #     return F.hinge_embedding_loss(input, target, self.margin, self.size_average)





data_file =""
queries_file = ""
net = SimpleRankNet(300,50,1)
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, mo0mentum=0.9)
data = PairWiseDataLoaer(data_file,queries_file)
data_loading = DataLoader(data,num_workers=4,shuffle=True,batch_size=5)
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i,batch in data_loading:
        inputs, labels = batch
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
