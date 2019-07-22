import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Read the data and process them
df = pd.read_csv('bike_day (3).csv')
X = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']]
y = df[['cnt']]
y = np.array(y)

#Use linear regression
model = LinearRegression()
model.fit(X, y)
r_sq = model.score(X, y)
y_pred = model.predict(X)
rmse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

#Print the output
print('coefficient of determination: ', r_sq)
print('intercept (b0): ', model.intercept_)
print('coefficient: ', model.coef_)
print('predicted response: ', y_pred, '\t')
print('root mean squared error: ', rmse)

#Plot the two y and y_pred values
x_index = df[['instant']]

plt.figure()
plt.plot(x_index, y, 'ro')
plt.plot(x_index, y_pred, 'g^')

plt.xlabel('index')
plt.ylabel('y')

#Plot feature importance

def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure((14, 8))
    plt.bar(pos, feature_importances[index_sorted], 'center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

plt.show()














