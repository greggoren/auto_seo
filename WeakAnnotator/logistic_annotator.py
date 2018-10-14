from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("coherence_final.csv",delimiter=",")
train,test = train_test_split(df, test_size=0.25)
x_train,y_train,x_test,y_test = train.iloc[:,1:-1],train.iloc[:,-1],test.iloc[:,1:-1],test.iloc[:,-1]
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)
print("accuracy is:",logisticRegr.score(x_test,y_test))
