from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences

text = """You can purchase dog waste bags online. Are there any low cost options? There are plenty of low cost alternatives to dog waste bags. Dog owners could use old newspapers, spare plastic bags, nappy sacks, or use a pooper-scooper and throw waste directly into the dog waste bin. I have seen bins with a picture of a dog on them. What does this mean? dog The council is installing new bins which allow for both refuse and dog waste. So when you see this dog on a bin, it means you can put your doggy bag in it. The new multi-use bins will be dotted around Ashford and the town centre, making it easier for dog owners to clean up after their pets and help keep the borough looking clean."""
print(retrieve_sentences(text))

# ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
# ASR_MONGO_PORT = 27017
# from pymongo import MongoClient
# client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
# db = client.asr16
# iterations = db.archive.distinct("iteration")
# sorted_iterations = sorted(iterations)
# print(sorted_iterations)