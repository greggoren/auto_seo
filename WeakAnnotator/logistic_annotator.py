from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("coherence.csv",delimiter=",")
# train,test = train_test_split(df, test_size=0.25)
# x_train,y_train,x_test,y_test = train.iloc[:,1:-1],train.iloc[:,-1],test.iloc[:,1:-1],test.iloc[:,-1]
logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train,y_train)
# print("accuracy is:",logisticRegr.score(x_test,y_test))
# print(logisticRegr.predict(x_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(logisticRegr, df.iloc[:,1:-1], df.iloc[:,-1], cv=5)
print("cv acc:")
print(scores.mean())
print("labels:")
print("1:",len(df[df.score==1]))
print("0:",len(df[df.score==0]))
