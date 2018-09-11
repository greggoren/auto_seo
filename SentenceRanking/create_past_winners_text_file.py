from Preprocess.preprocess import retrieve_ranked_lists,load_file
import params

doc_texts = load_file(params.trec_text_file)

f=open("past_winners_file","w")
for run_name in range(1,4):
    trec_file = "/home/greg/auto_seo/data/trec_file" + str(run_name)
    ranked_lists = retrieve_ranked_lists(trec_file)
    winners = {q:ranked_lists[q][0] for q in ranked_lists}
    for query in ranked_lists:
        f.write(query+"\t"+doc_texts[winners[query]]+"\n")
f.close()