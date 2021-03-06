from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
import params

doc_texts = load_file(params.trec_text_file)

f=open("past_winners_file_new_data04","w")
for run_name in range(1,4):
    trec_file = "/home/greg/auto_seo/data/trec_file" + str(run_name)
    ranked_lists = retrieve_ranked_lists(trec_file)
    winners = {q:ranked_lists[q][0] for q in ranked_lists}
    for query in ranked_lists:
        text = doc_texts[winners[query]].rstrip()
        sentences = retrieve_sentences(text)
        f.write(query+"45"+"@@@"+" ".join([a.replace("\n","")  for a in sentences])+"\n")
        f.write(query+"42"+"@@@"+" ".join([a.replace("\n","")  for a in sentences])+"\n")
f.close()

f=open("past_winners_file_new_data06","w")
for run_name in range(1,6):
    trec_file = "/home/greg/auto_seo/data/trec_file" + str(run_name)
    ranked_lists = retrieve_ranked_lists(trec_file)
    winners = {q:ranked_lists[q][0] for q in ranked_lists}
    for query in ranked_lists:
        text = doc_texts[winners[query]].rstrip()
        sentences = retrieve_sentences(text)
        f.write(query+"65"+"@@@"+" ".join([a.replace("\n","")  for a in sentences])+"\n")
        f.write(query+"62"+"@@@"+" ".join([a.replace("\n","")  for a in sentences])+"\n")
f.close()

f=open("new_data_queries","w")
with open("/home/greg/auto_seo/data/queris.txt") as file:
    for line in file:
        query = line.split(":")[0]
        query_text = line.split(":")[1].rstrip()
        rounds=["4","6"]
        ref_index = ["5","2"]
        for r in rounds:
            for ref in ref_index:
                key=r+ref
                f.write(query+key+":"+query_text+"\n")
f.close()