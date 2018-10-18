import torch.optim as optim
import torch
from NeuralNetRanking.feed_forward_net import SimpleRankNet
from NeuralNetRanking.pairwise_data import PairWiseDataLoaer
from torch.utils.data import DataLoader
from NeuralNetRanking.loss import NewHingeLoss
import os
import pickle
import operator
from CrossValidationUtils.evaluator import eval
from utils import run_bash_command
from torch.nn.modules.loss import MarginRankingLoss
import torch.cuda as cuda
def train_model(lr,momentum,labels_file,input_dir,batch_size,epochs,fold,p):
    net = SimpleRankNet(300, 150, 1,p)
    net = net.double()
    if cuda.is_available():
        print("cuda is on!!")
        net.cuda()

    criterion = MarginRankingLoss(margin=1)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    data = PairWiseDataLoaer(labels_file, input_dir)
    data_loading = DataLoader(data, num_workers=5, shuffle=True, batch_size=batch_size)
    epochs = epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(data_loading):
            inputs, labels = batch

            # forward + backward + optimize
            out1, out2 = net(inputs)
            optimizer.zero_grad()
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    models_dir = "models/"+str(fold)+"/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = "model_"+str(lr)+"_"+str(momentum)+"_"+str(batch_size)+"_"+str(epochs)
    torch.save(net,models_dir+model_name)
    # with open(models_dir+model_name,"wb") as model_file:
    #     pickle.dump(net,model_file)
    return net,models_dir+model_name



def load_object(file):
    with open(file,"rb") as example:
        return pickle.load(example)

def predict_folder_content(input_folder,model):
    results={}
    for file in os.listdir(input_folder):
        sample = load_object(input_folder + file)
        results[int(file)] = model(sample)[0].data[0].item()
    return results


def crossvalidation(folds_folder,number_of_folds,combination_name_indexes,qrels,summary_file):


    torch.multiprocessing.set_start_method("spawn")

    lrs = [0.01,0.001]
    batch_sizes = [5]
    epochs = [5,10,17]
    momentums = [0.9]
    dropouts = [0.2,0.5]
    scores={}
    models = {}
    evaluator = eval(metrics=["map","ndcg_cut.20","P.5","P.10"])
    test_trec_file = "NN_test_trec_file.txt"
    for fold in range(1,number_of_folds+1):
        print("in fold:",fold)
        models[fold]={}
        scores[fold]={}
        training_folder = folds_folder+str(fold)+"/train/"
        validation_folder = folds_folder+str(fold)+"/validation/"
        test_folder = folds_folder+str(fold)+"/test/"
        validation_results_folder = folds_folder+str(fold)+"/validation_results/"
        if not os.path.exists(validation_results_folder):
            os.makedirs(validation_results_folder)
        current_labels_file = "labels_fold_"+str(fold)+".pkl"
        for lr in lrs:
            for epoch in epochs:
                for momentum in momentums:
                    for batch_size in batch_sizes:
                        for p in dropouts:
                            model_name ="_".join((str(lr),str(epoch),str(momentum),str(batch_size)))
                            model,model_file = train_model(lr,momentum,current_labels_file,training_folder,batch_size,epoch,fold)
                            results = predict_folder_content(validation_folder,model)
                            trec_file_name = validation_results_folder+"NN_"+model_name+".txt"
                            evaluator.create_trec_eval_file_nn(results,combination_name_indexes["val"][fold],trec_file_name)
                            score = evaluator.run_trec_eval(trec_file_name,qrels)
                            scores[fold][model_name] = float(score)
                            models[fold][model_name]=model_file
        best_model = max(scores[fold].items(), key=operator.itemgetter(1))[0]
        print("chosen model on fold",fold,":",best_model)
        # test_model = load_object(models[fold][best_model])
        test_model = torch.load(models[fold][best_model])
        results = predict_folder_content(test_folder,test_model)
        evaluator.create_trec_eval_file_nn(results, combination_name_indexes["test"][fold], test_trec_file,True)
    final_trec_file = evaluator.order_trec_file(test_trec_file)
    run_bash_command("rm "+test_trec_file)
    evaluator.run_trec_eval_on_test(summary_file=summary_file,qrels=qrels,method="NN",trec_file=final_trec_file)




if __name__=="__main__":
    folds_folder="folds/"
    number_of_folds=5
    combination_name_indexes=load_object("test_names.pkl")
    print(combination_name_indexes["val"][1][0])
    qrels="/home/greg/auto_seo/SentenceRanking/labels_final1"
    summary_file="NN_cv_summary_with_dropout.tex"
    print("starting CV")
    crossvalidation(folds_folder,number_of_folds,combination_name_indexes,qrels,summary_file)









