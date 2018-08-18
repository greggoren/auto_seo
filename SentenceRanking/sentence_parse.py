from Preprocess.preprocess import retrieve_sentences

def map_sentences(document_texts,winners):
    sentence_map={}
    for query in winners:
        sentence_map[query]={}
        winner = winners[query].replace("EPOCH","ROUND")
        text = document_texts[winner]
        sentences = retrieve_sentences(text)
        index = 1
        for sentence in sentences:
            sentence_map[query][winner+"_"+str(index)]=sentence
            index+=1
    return sentence_map

def create_lists(scores):
    lists = {}
    for doc in scores:
        query = doc.split("-")[2]
        if not lists.get(query,False):
            lists[query]={}
        lists[query][doc]=scores[doc]
    results = {}
    for query in lists:
        results[query]=sorted(sorted(list(lists[query].keys()),key=lambda x:x),key=lambda x:lists[query][x],reverse=True)
    return results

def add_labeles():

    label_file = open("/home/greg/auto_seo/SentenceRanking/labels")
    labels = {line.split("\t")[0]:line.split("\t")[1].replace("\n","") for line in label_file}
    label_file.close()
    new_features = open("sentenceFeaturesFinal","w")
    with open("/home/greg/auto_seo/SentenceRanking/sentenceFeatures") as features:
        for line in features:
            splited = line.split()
            label = labels[splited[-1].rstrip()]
            new_line = label+" "+" ".join(splited[1:])
            new_features.write(new_line+"\n")
        new_features.close()