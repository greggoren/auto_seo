from Preprocess.preprocess import retrieve_ranked_lists,load_file
from SentenceRanking.sentence_parse import  map_set_of_sentences
from Preprocess.preprocess import retrieve_sentences
import params
import sys





ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
a_doc_texts = load_file(params.trec_text_file)
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc]=a_doc_texts[doc]
sentence_map=map_set_of_sentences(doc_texts,winner_docs)
summaries = {}
sentence_data_file = open("senetces_add_remove", "w")
for query in sentence_map:
    sys.stdout.flush()
    reference_doc = reference_docs[query].replace("EPOCH","ROUND")
    reference_text = doc_texts[reference_doc]
    reference_sentences = retrieve_sentences(reference_text)

    for sentence in sentence_map[query]:
        r_index = 1
        new_sentence = sentence_map[query][sentence].replace("\n", "")
        if not new_sentence:
            continue

        for reference_sentence in reference_sentences:
            run_name = sentence+"_"+str(r_index)

            reference_sentence=reference_sentence.replace("\n", "")
            if not reference_sentence:
                continue
            modified_doc=reference_doc+"\n"+new_sentence
            summaries[reference_doc]=modified_doc
            sentence_data_file.write(run_name + "\t" + new_sentence.rstrip() + "\t" + reference_sentence.rstrip() + "\n")
            r_index+=1


sentence_data_file.close()