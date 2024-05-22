import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('C:/Users/User/Desktop/200 PYTHON PROJECT CHALLENGE/data/Day14/Predict iPhone Price/iphone_price.csv')
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print("iphone price:")
print(model.predict([[14]]))
# print(data.head)
plt.scatter(data['version'], data['price'])
plt.show()