import pandas as pd
from main import Adaboost
test1 = Adaboost(base=1)
test1.fit('data.csv', 'targets.csv')
testy1 = test1.predict('data.csv')
Y = pd.read_csv('targets.csv', header=None)
print((testy1 == Y.values[:, 0]).mean())
