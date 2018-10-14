from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
df = pd.read_csv("coherence.csv",delimiter=",")
kf = KFold(n_splits=5)
split = kf.split(df)
n_estimators= [150,200,500,1000,10000]
n_depth = [1,2,3,4,5,10,100]

for e in n_estimators:
    for d in n_depth:


        print("fitting on params: max_depth=",d,"n_estimators=",e)
        gbrtRegr = GradientBoostingRegressor(n_estimators=e,max_depth=d)
        r2=[]
        mse = []
        for train,test in split:
            gbrtRegr.fit(df.iloc[train,1:-1],df.iloc[train, -1])
            predictions = gbrtRegr.predict(df.iloc[test,1:-1])
            r2.append(r2_score(predictions,df.iloc[test,-1]))
            mse.append(mean_squared_error(predictions,df.iloc[test,-1]))

        print("cv r2:")
        print(np.mean(r2))
        print("cv mse:")
        print(np.mean(mse))



