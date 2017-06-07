import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


df = pd.read_fwf('brain_body.txt')#, delimiter = ' ', encoding = 'utf-8-sig')

x = df[['Brain']]
y = df[['Body']]

reg = linear_model.LinearRegression()

reg.fit(x,y)

plt.scatter(x,y)
plt.plot(x,reg.predict(x))
plt.show()