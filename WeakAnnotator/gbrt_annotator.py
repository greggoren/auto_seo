from sklearn.linear_model import LogisticRegression
import pandas as pd
from random import shuffle
from sklearn.model_selection import cross_val_score

df = pd.read_csv("coherence.csv",delimiter=",")
indexes = df.index[df.score == 0].tolist()
shuffle(indexes)
indexes_sliced = indexes[:84]
df_1 = df[df.score==1]
df_0 = df.iloc[indexes_sliced]
frames = [df_0,df_1]
df_unified = pd.concat(frames)

logisticRegr = LogisticRegression(class_weight='balanced')

scores = cross_val_score(logisticRegr, df_unified.iloc[:,1:-1], df_unified.iloc[:,-1], cv=5)
print("cv acc:")
print(scores.mean())



