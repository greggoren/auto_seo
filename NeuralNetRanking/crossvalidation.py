import torch.optim as optim
from NeuralNetRanking.feed_forward_net import SimpleRankNet
from NeuralNetRanking.pairwise_data import PairWiseDataLoaer
from torch.utils.data import DataLoader
from NeuralNetRanking.loss import NewHingeLoss
import os
import pickle

def train_model(lr,momentum,labels_file,input_dir,batch_size,epochs,fold):
    net = SimpleRankNet(300, 50, 1)
    net = net.double()
    net.cuda()
    criterion = NewHingeLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    data = PairWiseDataLoaer(labels_file, input_dir)
    data_loading = DataLoader(data, num_workers=4, shuffle=True, batch_size=batch_size)
    epochs = epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(data_loading):
            inputs, labels = batch
            optimizer.zero_grad()

            # forward + backward + optimize
            out1, out2 = net(inputs)
            loss = criterion(out1, out1, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    models_dir = "models/"+str(fold)+"/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = "model_"+str(lr)+"_"+str(momentum)+"_"+str(batch_size)+"_"+str(epochs)
    with open(models_dir+model_name,"wb") as model_file:
        pickle.dump(net,model_file)







