from sklearn.linear_model import LogisticRegression
import pandas as pd
from random import shuffle
from sklearn.model_selection import cross_val_score

df = pd.read_csv("coherence.csv",delimiter=",")
indexes = df.index[df.score == 0].tolist()


logisticRegr = LogisticRegression(class_weight='balanced')

scores = cross_val_score(logisticRegr, df.iloc[:,1:-1], df.iloc[:,-1], cv=5,scoring="accuracy")
print("cv acc:")
print(scores.mean())

scores = cross_val_score(logisticRegr, df.iloc[:,1:-1], df.iloc[:,-1], cv=5,scoring="f1")
print("cv f1:")
print(scores.mean())

# train,test = train_test_split(df, test_size=0.25)
# x_train,y_train,x_test,y_test = train.iloc[:,1:-1],train.iloc[:,-1],test.iloc[:,1:-1],test.iloc[:,-1]
# logisticRegr.fit(x_train,y_train)
# print("accuracy is on one shot :",logisticRegr.score(x_test,y_test))
# print(logisticRegr.predict(x_test))
