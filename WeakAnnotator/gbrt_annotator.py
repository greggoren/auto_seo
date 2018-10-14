from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
df = pd.read_csv("coherence.csv",delimiter=",")
kf = KFold(n_splits=5)

n_estimators= [100,150,200,500,1000,10000]
n_depth = [1,2,3,4,5,10,100]
results = []
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
            mse.append(mean_squared_error(df.iloc[test,-1],predictions))

        score = np.mean(mse)
        results.append([(e,d),score])

print("final results")
print(sorted(results,key=lambda x:x[1]))


