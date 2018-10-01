import operator
import os
from utils import run_bash_command


class model_handler_LambdaMart:

    def __init__(self, trees, leaves):
        self.leaves_param = leaves
        self.trees_param = trees
        self.home_path = "/home/greg/"
        self.code_base_path = self.home_path+"auto_seo/"
        self.java_path = self.home_path+"jdk1.8.0_181/bin/java"
        self.jar_path = self.home_path+"SEO_CODE/model_running/RankLib.jar"
        self.model_base_path=self.code_base_path+"/CrossValidationUtils/lm_models/"
        self.query_to_fold_index={}
        self.chosen_model_per_fold ={}


    def set_queries_to_folds(self,queries,test_indices,fold):
        set_of_queries = set(queries[test_indices])
        tmp = {a:fold for a in set_of_queries}
        self.query_to_fold_index.update(tmp)

    def create_model_LambdaMart(self, number_of_trees, number_of_leaves, train_file,fold,test=False):

        if test:
            add="test"
        else:
            add=""
        # model_path = self.model_base_path+str(fold) +"/" + add +'model_' + str(number_of_trees) + "_" + str(number_of_leaves)
        model_path = "lm_models/"+str(fold) +"/"+ add +'model_' + str(number_of_trees) + "_" + str(number_of_leaves)

        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        command = self.java_path + ' -jar ' + self.jar_path + ' -train ' + train_file + ' -ranker 6    -metric2t NDCG@20' \
                                                                                        ' -tree ' + str(number_of_trees) + ' -leaf ' + str(number_of_leaves) +' -save ' +model_path
        print("command = ", command)
        run_bash_command(command)
        return model_path

    def run_model(self,test_file,fold,trees,leaves,model_path):#TODO:add to main functionality + test file
        # score_file = self.code_base_path+"lm_scores/"+str(fold)+"/score" + str(trees)+"_"+str(leaves)
        score_file = "lm_score/"+str(fold)+"/score" + str(trees)+"_"+str(leaves)
        if not os.path.exists(os.path.dirname(score_file)):
            os.makedirs(os.path.dirname(score_file))
        run_bash_command('touch '+score_file)
        command = self.java_path + " -jar " + self.jar_path + " -load " + model_path + " -rank " + test_file + " -score " + score_file
        run_bash_command(command)
        return score_file
    #
    def run_model_on_test(self,test_file,fold):
        trees,leaves = self.chosen_model_per_fold[fold]
        score_file = "lm_score/" + str(fold) + "/score" + str(trees) + "_" + str(leaves)
        if not os.path.exists(os.path.dirname(score_file)):
            os.makedirs(os.path.dirname(score_file))
        run_bash_command('touch ' + score_file)

        model_path ="lm_models/"+str(fold) +"/model_" + str(trees) + "_" + str(leaves)
        run_bash_command('touch '+score_file)
        command = self.java_path + " -jar " + self.jar_path + " -load " + model_path + " -rank " + test_file + " -score " + score_file
        run_bash_command(command)
        return score_file


    def retrieve_scores(self,test_indices,score_file):
        with open(score_file) as scores:
            results={test_indices[i]:score.split()[2].rstrip() for i,score in enumerate(scores)}
            return results



    # def fit_model_on_train_set_and_choose_best_for_competition(self,train_file,test_file,validation_indices,queries,evaluator):
    #     evaluator.empty_validation_files()
    #     scores={}
    #     for trees_number in self.trees_param:
    #         for leaf_number in self.leaves_param:
    #             print("fitting model on trees=", trees_number,"leaves = ",leaf_number)
    #             self.create_model_LambdaMart(trees_number,leaf_number,train_file,params.qrels)
    #             score_file=self.run_model(test_file,trees_number,leaf_number)
    #             results = self.retrieve_scores(validation_indices,score_file)
    #             trec_file=evaluator.create_trec_eval_file(validation_indices, queries, results, "model_"+str(trees_number)+"_"+str(leaf_number), validation=True)
    #             score = evaluator.run_trec_eval(trec_file)
    #             scores[(trees_number,leaf_number)] = score
    #     trees,leaves=max(scores.items(), key=operator.itemgetter(1))[0]
    #     print("the chosen model is trees=",trees," leaves=",leaves)
    #     self.create_model_LambdaMart(trees,leaves,params.data_set_file,params.qrels,True)

    def fit_model_on_train_set_and_choose_best(self,train_file,test_file,validation_indices,queries,fold,evaluator,qrels):
        print("fitting models on fold",fold)
        scores={}
        for trees_number in self.trees_param:
            for leaf_number in self.leaves_param:
                model_path= self.create_model_LambdaMart(trees_number,leaf_number,train_file,fold)
                score_file = self.run_model(test_file,fold,trees_number,leaf_number,model_path)
                results = self.retrieve_scores(validation_indices,score_file)
                trec_file=evaluator.create_trec_eval_file(validation_indices,queries,results,"_".join([str(a) for a in (trees_number,leaf_number)]),"lm",fold,True)
                score = evaluator.run_trec_eval(trec_file,qrels)
                scores[((trees_number,leaf_number))] = score
        trees, leaves = max(scores.items(), key=operator.itemgetter(1))[0]
        print("the chosen model is trees=", trees, " leaves=", leaves)
        self.chosen_model_per_fold[fold]=(trees,leaves)