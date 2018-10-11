import torch.optim as optim
from NeuralNetRanking.feed_forward_net import SimpleRankNet
from NeuralNetRanking.pairwise_data import PairWiseDataLoaer
from torch.utils.data import DataLoader
from NeuralNetRanking.loss import NewHingeLoss





data_file ="/home/greg/auto_seo/NeuralNetRanking/new_sentences_add_remove"
queries_file = "/home/greg/auto_seo/data/queris.txt"
net = SimpleRankNet(300,50,1)
print(net)
criterion = NewHingeLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
data = PairWiseDataLoaer(data_file,queries_file)
data_loading = DataLoader(data,num_workers=4,shuffle=True,batch_size=5)
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i,batch in enumerate(data_loading):
        inputs, labels = batch
        optimizer.zero_grad()

        # forward + backward + optimize
        out1,out2 = net(inputs)
        loss = criterion(out1,out1, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
