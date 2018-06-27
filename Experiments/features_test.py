import pickle


f = open("dic.pickle",'r')
test=pickle.load(f)
f.close()
if "clueweb09-en0003-04-23098" in test:
    print("OK")
else:
    print("no!")