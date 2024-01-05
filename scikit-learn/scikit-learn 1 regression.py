# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:42:51 2024
as part of Python for Data Science and Machine Learning Bootcamp 
@author: kagan
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Loading and exploring the dataset
us_housing = pd.read_csv("C:/python/ML/USA_Housing.csv")
us_housing.head()
us_housing.info()
us_housing.describe()
us_housing.columns

# Visualizing the data
sns.pairplot(us_housing)
sns.displot(us_housing["Price"])
sns.heatmap(us_housing.corr(), annot=True)

# Splitting features and target variable
X = us_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
y = us_housing['Price']

# Splitting data into training and testing sets. We set the test set to 40%. 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Creating a linear regression model object
lm_model = LinearRegression()

# Training the linear regression model
lm_model.fit(x_train, y_train)

# Checking the coefficients of the model
print(lm_model.intercept_)
coefficients = pd.DataFrame(lm_model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# Making predictions
predictions = lm_model.predict(x_test)

# Visualization of predictions vs actual values
plt.scatter(y_test, predictions)
sns.distplot((y_test - predictions), bins=50)

# Evaluating the model
MAE = metrics.mean_absolute_error(y_test, predictions)
MSE = metrics.mean_squared_error(y_test, predictions)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))

print('MAE:', round(MAE,2))
print('MSE:', round(MSE,2))
print('RMSE:', round(RMSE,2))

