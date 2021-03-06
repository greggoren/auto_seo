import torch.optim as optim
from NeuralNetRanking.feed_forward_net import SimpleRankNet
from NeuralNetRanking.pairwise_data import PairWiseDataLoaer
from torch.utils.data import DataLoader
import torch
from torch.nn.modules.loss import MarginRankingLoss
import torch.cuda as cuda


if __name__=="__main__":
    torch.multiprocessing.set_start_method("spawn")


    data_file ="/home/greg/auto_seo/NeuralNetRanking/new_sentences_add_remove"
    queries_file = "/home/greg/auto_seo/data/queris.txt"
    net = SimpleRankNet(300,150,1)
    net = net.double()
    input_dir = "input/"
    if cuda.is_available():
        print("cuda bitch!")
        net.cuda()
        input_dir="input_gpu/"
    # use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
    #                                              if torch.cuda.is_available() and x
    #                                              else torch.FloatTensor)
    print(net)
    criterion = MarginRankingLoss(margin=1)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    data = PairWiseDataLoaer("labels/labels.pkl",input_dir)
    print("in data loading")
    data_loading = DataLoader(data,num_workers=5,shuffle=True,batch_size=5)
    epochs = 1000
    for epoch in range(epochs):
        running_loss = 0.0
        for i,batch in enumerate(data_loading):
            inputs, labels = batch


            # forward + backward + optimize
            out1,out2 = net(inputs)

            optimizer.zero_grad()
            loss = criterion(out1,out2, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000),flush=True)
                running_loss = 0.0
                print(out1)
                print(out2)
