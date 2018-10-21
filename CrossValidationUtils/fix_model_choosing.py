from CrossValidationUtils.evaluator import eval
trees=[500,250]
leaves = [5,10,25,50]
evaluator = eval(metrics=[""])
qrels = "/home/greg/auto_seo/CrossValidationUtils/mq_track_qrels"
base_folder = "/home/greg/auto_seo/CrossValidationUtils/lm_validation/0/"
for tree in trees:
    for leaf in leaves:
        file_name = base_folder+"trec_file_model_"+str(tree)+"_"+str(leaf)+".txt"
        tmp_file = file_name+"_tmp"
        f = open(tmp_file,"w")
        with open(file_name) as file:
            for line in file:
                new_line_splits = line.split()
                new_line = " ".join([new_line_splits[0],new_line_splits[1],new_line_splits[2],new_line_splits[3],new_line_splits[6],new_line_splits[7]])+'\n'
                f.write(new_line)
        f.close()
        print("on ",tmp_file)
        final_trec_score = evaluator.run_trec_eval(tmp_file,qrels)
