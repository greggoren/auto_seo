import shutil
import subprocess
import os

class eval:




    def __init__(self):
        self.metrics = ["map","ndcg_cut.20","P.5","P.10"]
        self.validation_metric = "ndcg_cut.20"
        self.doc_name_index = {}

    def remove_score_file_from_last_run(self):
        if os._exists("scores"):
            os.remove("scores")

    def create_trec_eval_file(self, test_indices, queries, results,model,validation=None):#TODO: need to sort file via unix command
        if validation is not None:
            trec_file = "validation/trec_file_"+model+".txt"
            if not os.path.exists(os.path.dirname(trec_file)):
                os.makedirs(os.path.dirname(trec_file))

        else:
            trec_file = "scores"
        trec_file_access = open(trec_file,'a')
        for index in test_indices:
            trec_file_access.write(self.set_qid_for_trec(queries[index])+" Q0 "+self.doc_name_index[index]+" "+str(0)+" "+str(results[index])+" seo\n")
        trec_file_access.close()
        return trec_file



    def order_trec_file(self,trec_file):
        final = trec_file.replace(".txt","")
        command = "sort -k1,1 -k5nr -k2,1 "+trec_file+" > "+final
        for line in self.run_command(command):
            print(line)
        return final

    def run_command(self, command):
        p = subprocess.Popen(command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=True)
        return iter(p.stdout.readline, b'')

    def run_trec_eval(self, score_file,qrels):
        command = "./trec_eval -m " + self.validation_metric +" "+ qrels+" " + score_file
        for output_line in self.run_command(command):
            print("output line=",output_line)
            score = output_line.split()[-1].rstrip()
            break
        return score

    def empty_validation_files(self):
        try:
            shutil.rmtree("validation")
        except:
            print("no validation folder")


    def run_trec_eval_on_test(self,qrels,trec_file=None):
        if trec_file is None:
            trec_file="scores"
        score_data = []
        print("last stats:")
        for metric in self.metrics:
            command = "./trec_eval -m " + metric + " "+qrels+" " + trec_file
            for output_line in self.run_command(command):
                print(metric,output_line)
                score = output_line.split()[-1].rstrip()
                score_data.append((metric, str(score)))
        summary_file = open("summary", 'w')
        summary_file.write("METRIC\tSCORE\n")
        for score_record in score_data:
            next_line = score_record[0] + "\t" + score_record[1] + "\n"
            summary_file.write(next_line)
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