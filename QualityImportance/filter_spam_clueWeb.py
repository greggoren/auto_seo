def retrieve_non_spam(non_spam):
    docs = []
    with open(non_spam) as file:
        for line in file:
            docs.append(line.split(" # ")[1].split("\n")[0])
    return docs

def create_asr_non_spam_fetures(asr,non_spam_doc):
    f = open("features_filtered","w")
    with open(asr) as file:
        for line in file:
            name = line.split(" # ")[1].split("\n")[0]
            if name in non_spam_doc:
                f.write(line)
        f.close()


docs =retrieve_non_spam("featuresF")
create_asr_non_spam_fetures("Quality_Features",docs)
