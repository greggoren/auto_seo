from scipy.stats import kendalltau

a = ["bn","b2","bs","bs2","bs1"]
b = ["b2","bn","bs","bs2","bs1"]
print(kendalltau(a,b)[0])