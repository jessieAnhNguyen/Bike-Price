import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('bike_hour (2).csv')
data = df.values
x = df[['season', 'yr', 'mnth', 'holiday', 'hr', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]
x = np.array(x)
print(x.size)
"""
x = x.reshape(731, 14)
for i in range(1, 14, 1):
    for j in range (1, 731, 1):
        x[i][j] = (x[i][j] - np.mean(x[i])) / (np.max(x[i] - np.min(x[i])))
        x[i][j] = x[i].normalize();


x1 = df[['season']]
x1 = np.array(x1)
for i in range(1, x1.size, 1):
    x1[i] = (x1[i] - np.mean(x1)) / (np.max(x1) - np.min(x1))

x2 = df[['yr']]
x2 = np.array(x2)
for i in range(1, x2.size, 1):
    x2[i] = (x2[i] - np.mean(x2)) / (np.max(x2) - np.min(x2))

"""

y = df[['cnt']]
y = np.array(y)

model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
y_pred = model.predict(x)
rmse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)


print('coefficient of determination: ', r_sq)
print('intercept (b0): ', model.intercept_)
print('coefficient: ', model.coef_)
print('predicted response: ', y_pred, '\t')
print('root mean squared error: ', rmse)


x_index = df[['instant']]

plt.figure()
plt.plot(x_index, y, 'ro')
plt.plot(x_index, y_pred, 'g^')

plt.xlabel('index')
plt.ylabel('y')

"""
plt.scatter(y, y_pred)
plt.xlabel('y')
plt.ylabel('y_pred')
"""

plt.show()














