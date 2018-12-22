import shutil
import subprocess
import os
from utils import run_bash_command,run_command

class eval:




    def __init__(self,metrics):
        self.metrics = metrics
        self.validation_metric = "ndcg_cut.5"
        self.doc_name_index = {}

    def remove_score_file_from_last_run(self,method):
        file = method+"_scores"
        if os._exists(file):
            os.remove(file)

    def run_trec_eval_by_query(self,qrels, trec_file):
        score_data = {}
        print("last stats:")
        for metric in self.metrics:
            command = "./trec_eval -q -m " + metric + " " + qrels + " " + trec_file
            score_data[metric] = []
            for line in run_command(command):
                if len(line.split()) <= 1:
                    break
                if str(line.split()[1]).replace("b'", "").replace("'", "") == "all":
                    break
                score = float(line.split()[2].rstrip())
                # query = str(line.split()[1])
                # query = query.replace("b'", "")
                # query = query.replace("'", "")
                score = str(score).replace("b'", "")
                score = score.replace("'", "")
                score_data[metric].append(float(score))
        return score_data


    def create_trec_eval_file(self, test_indices, queries, results,model,method,fold,validation=None):
        if validation is not None:
            trec_file = method+"_validation/"+str(fold)+"/trec_file_"+model+".txt"
            if not os.path.exists(os.path.dirname(trec_file)):
                os.makedirs(os.path.dirname(trec_file))

        else:
            trec_file = method+"_scores.txt"
        trec_file_access = open(trec_file,'a')
        for index in test_indices:
            trec_file_access.write(self.set_qid_for_trec(queries[index])+" Q0 "+self.doc_name_index[index]+" "+str(0)+" "+str(results[index])+" seo\n")
        trec_file_access.close()
        return trec_file



    def create_trec_eval_file_nn(self,results,comb_index,file_name,test = False):
        access = "w"
        if test:
            access = "a"
        f = open(file_name,access)

        for index in results:
            combination = comb_index[index]
            query = combination.split("-")[2]
            line = query+" Q0 "+combination+" "+str(0)+" "+str(results[index])+" seo\n"
            f.write(line)





    def order_trec_file(self,trec_file):
        final = trec_file.replace(".txt","")
        command = "sort -k1,1 -k5nr -k2,1 "+trec_file+" > "+final
        print(command)
        run_bash_command(command)
        return final

    def run_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True)
        return iter(p.stdout.readline, b'')

    def run_trec_eval(self, score_file,qrels):
        score_file_final = self.order_trec_file(score_file)
        command = "./trec_eval -m " + self.validation_metric +" "+ qrels+" " + score_file_final
        for output_line in self.run_command(command):
            print("output line=",output_line)
            score = output_line.split()[-1].rstrip()
            break
        return score

    def empty_validation_files(self,method):
        try:
            shutil.rmtree(method+"_validation")
        except:
            print("no validation folder")


    def run_trec_eval_on_test(self,qrels,summary_file,method,trec_file=None,increase_stats=False):
        if trec_file is None:
            trec_file=method+"_scores"
        score_data = []
        print("last stats:")
        for metric in self.metrics:
            command = "./trec_eval -m " + metric + " "+qrels+" " + trec_file
            for output_line in self.run_command(command):
                print(metric,output_line)
                score = output_line.split()[-1].rstrip()
                score = str(score).replace("b'","")
                score = score.replace("'","")
                score_data.append((metric, str(score)))
        summary_file = open(summary_file, 'w')
        if not increase_stats:
            cols = "c|"*len(self.metrics)
        else:
            cols = "c|" * (len(self.metrics)+len(increase_stats))
        cols = "|"+cols
        summary_file.write("\\begin{tabular}{"+cols+"}\n")
        if not increase_stats:
            next_line = " & ".join([s[0] for s in score_data])+"\\\\ \n"
        else:
            next_line = " & ".join([s[0] for s in score_data])+" & "+" & ".join(["1","2","5"])+"\n"

        summary_file.write(next_line)
        if not increase_stats:
            next_line = " & ".join([s[1] for s in score_data]) + "\n"
        else:
            next_line = " & ".join([s[1] for s in score_data]) +" & "+" & ".join([str(round(increase_stats[j]),4) for j in [1,2,5]])+ + "\\\\ \n"
        summary_file.write(next_line)
        summary_file.write("\\end{tabular}")
        summary_file.close()

    def create_index_to_doc_name_dict(self,features):
        index =0
        with open(features) as ds:
            for line in ds:
                rec = line.split("# ")
                doc_name = rec[1].rstrip()
                self.doc_name_index[index]=doc_name
                index+=1

    def set_qid_for_trec(self,query):
        if query < 10:
            qid = "00" + str(query)
        elif query < 100:
            qid = "0" + str(query)
        else:
            qid = str(query)
        return qid

    def create_qrels_file(self,X,y,queries):
        print("creating qrels file")
        qrels = open("qrels",'w')
        for i in range(len(X)):
            qrels.write(self.set_qid_for_trec(queries[i]) + " 0 " + self.doc_name_index[i] + " " + str(int(y[i])) + "\n")
        qrels.close()
        print("qrels file ended")