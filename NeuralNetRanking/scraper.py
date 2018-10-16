import operator
y ={"a":1,"b":2,"c":3}
a = max(y.items(),key=operator.itemgetter(1))[0]
print(a)