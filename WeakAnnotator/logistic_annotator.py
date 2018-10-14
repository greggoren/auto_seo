from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("coherence_final.csv",delimiter=",")
x_train, x_test, y_train, y_test = train_test_split(df[:,1:], df[:,-1], test_size=0.25, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train,y_train)
print("accuracy is:",logisticRegr.score(x_test,y_test))
