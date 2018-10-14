from sklearn.linear_model import LogisticRegression
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv("coherence.csv",delimiter=",")
indexes = df.index[df.score == 0].tolist()
shuffle(indexes)
indexes_sliced = indexes[:84]
# df_1 = df[df.score==1]
# df_0 = df.iloc[indexes]
# frames = [df_0,df_1]
# df_unified = pd.concat(frames)


logisticRegr = LogisticRegression()



scores = cross_val_score(logisticRegr, df.iloc[:,1:-1], df.iloc[:,-1], cv=5)
print("cv acc:")
print(scores.mean())



train,test = train_test_split(df, test_size=0.25)
x_train,y_train,x_test,y_test = train.iloc[:,1:-1],train.iloc[:,-1],test.iloc[:,1:-1],test.iloc[:,-1]
logisticRegr.fit(x_train,y_train,sample_weight="auto")
print("accuracy is on one shot :",logisticRegr.score(x_test,y_test))
print(logisticRegr.predict(x_test))
