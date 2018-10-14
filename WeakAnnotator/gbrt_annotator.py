from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
df = pd.read_csv("coherence.csv",delimiter=",")
kf = KFold(n_splits=5)

n_estimators= [150,200,500,1000,10000]
n_depth = [1,2,3,4,5,10,100]

for e in n_estimators:
    for d in n_depth:


        print("fitting on params: max_depth=",d,"n_estimators=",e)
        gbrtRegr = GradientBoostingRegressor(n_estimators=e, max_depth=d)
        split = kf.split(df)
        r2=[]
        mse = []
        for train,test in split:

            gbrtRegr.fit(df.iloc[train,1:-1],df.iloc[train, -1])
            predictions = gbrtRegr.predict(df.iloc[test,1:-1])
            print(predictions)
            r2.append(r2_score(df.iloc[test,-1],predictions))
            mse.append(mean_squared_error(df.iloc[test,-1],predictions))

        print("cv r2:")
        print(np.mean(r2))
        print("cv mse:")
        print(np.mean(mse))



