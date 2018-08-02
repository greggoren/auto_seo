# def retrieve_non_spam(non_spam):
#     docs = []
#     with open(non_spam) as file:
#         for line in file:
#             docs.append(line.split(" # ")[1].split("\n")[0])
#     return docs
#
# def create_asr_non_spam_fetures(asr,non_spam_doc):
#     f = open("features_filtered","w")
#     with open(asr) as file:
#         for line in file:
#             name = line.split(" # ")[1].split("\n")[0]
#             if name in non_spam_doc:
#                 f.write(line)
#         f.close()
#

# docs =retrieve_non_spam("featuresF")
# create_asr_non_spam_fetures("Quality_Features",docs)


def get_waterloo_score(waterloo_file):
    dict = {}
    with open(waterloo_file) as file:
        for line in file:
            doc_id = line.split(" # ")[1]
            waterloo = line.split("26:")[1].split(" ")[0]
            dict[doc_id]=waterloo
        return dict

def create_file(base,dic):
    f = open("features_with_waterloo",'w')
    with open(base) as file:
        for line in file:
            features,doc = line.split(" # ")[0],line.split(" # ")[1]
            waterloo = dic[doc]
            new_features = features+" 34:"+waterloo
            new_line = new_features+" # "+doc
            f.write(new_line)
        f.close()


d = get_waterloo_score("ClueWeb09Extra")
create_file("Quality_features",d)
