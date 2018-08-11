from Preprocess.preprocess import retrieve_sentences

def map_sentences(document_texts,winners):
    sentence_map={}
    for query in winners:
        sentence_map[query]={}
        winner = winners[query]
        text = document_texts[winner]
        sentences = retrieve_sentences(text)
        index = 1
        for sentence in sentences:
            sentence_map[query][winner+str(index)]=sentence
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
