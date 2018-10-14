from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from random import shuffle
from sklearn.model_selection import cross_val_score

df = pd.read_csv("coherence.csv",delimiter=",")
# indexes = df.index[df.score == 0].tolist()
# shuffle(indexes)
# indexes_sliced = indexes[:84]
# df_1 = df[df.score==1]
# df_0 = df.iloc[indexes_sliced]
# frames = [df_0,df_1]
# df_unified = pd.concat(frames)

gbrtRegr = GradientBoostingRegressor()

scores = cross_val_score(gbrtRegr, df.iloc[:, 1:-1], df.iloc[:, -1], cv=5,scoring="r2")
print("cv r2:")
print(scores.mean())


scores = cross_val_score(gbrtRegr, df.iloc[:, 1:-1], df.iloc[:, -1], cv=5,scoring="neg_mean_squared_error")
print("cv mse:")
print(scores.mean())


n_estimators= [150,200,500,1000,10000]
n_depth = [1,2,3,4,5,10,100]

for e in n_estimators:
    for d in n_depth:
        print("fitting on params: max_depth=",d,"n_estimators=",e)
        gbrtRegr = GradientBoostingRegressor(n_estimators=e,max_depth=d)

        scores = cross_val_score(gbrtRegr, df.iloc[:, 1:-1], df.iloc[:, -1], cv=5,scoring="r2")
        print("cv r2:")
        print(scores.mean())


        scores = cross_val_score(gbrtRegr, df.iloc[:, 1:-1], df.iloc[:, -1], cv=5,scoring="neg_mean_squared_error")
        print("cv mse:")
        print(scores.mean())
