# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:06:42 2020

@author: phamk
"""



import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Load data
import os
housing = pd.read_csv('housing.csv')

#Data understanding
print(housing.head(3))
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())

# Another quick way is visualize the data with graph
%matplotlib inline 
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()

#Some problems in histogram
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# Find the relate between housing price and location
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5, 
            s=housing['population'] / 50, label='population', figsize=(12, 8),
            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

#Compute the correlation matrix between attributes
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values())

#Cleaning data
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)

#Attributes combination
housing['room_per_household'] = housing['total_rooms'] / housing['households']
housing['bedroom_per_household'] = housing['total_bedrooms'] / housing['households']
housing['population_per_household'] = housing['population'] / housing['households']
print(housing.head(3))

corr_matrix = housing.corr()

print(corr_matrix['median_house_value'].sort_values())

#Handing the text to categorical attributes
# Pandas get dumies method
housing_dummies = pd.get_dummies(housing, prefix=['ocean_proximity'])
print(housing_dummies.head(3))

#Feature scaling
import numpy as np
tmp = np.array([1, 2, 3, 4, 4, 5, 67, 2, 21, 54])

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(tmp.reshape(-1, 1))
min_max_scaler.transform([[1, 3, 2]])

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(tmp.reshape(-1, 1))
std_scaler.transform([[1, 3, 2]])

# Transform data
scaler = MinMaxScaler()
housing_features = housing_dummies.drop('median_house_value', axis=1)
X_data = scaler.fit_transform(housing_features)

y_data = np.array(housing_dummies['median_house_value'])
y_data = y_data.reshape(len(y_data), 1)

# Train test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
X_train.shape
y_test.shape
X_train[0]

# Placeholder for pass data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

n_dim = 16
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# Trainable weights
W = tf.Variable(tf.random_uniform([n_dim, 1]), dtype=tf.float32)
b = tf.Variable(tf.zeros(1, dtype = tf.float32))

pred = tf.add(tf.matmul(X, W), b)

mse_loss = tf.reduce_mean(tf.square(Y - pred))

loss = tf.sqrt(mse_loss)

learning_rate = 300

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

# Training
sess.run(init)

epochs = 50000
for epoch in range(epochs):
    
    sess.run(optimizer, feed_dict={X: X_train, Y: y_train})
    
    train_loss = sess.run(loss, feed_dict={X : X_train, Y: y_train})
    test_loss = sess.run(loss, feed_dict={X : X_test, Y: y_test})
 
    if epoch % 1000 == 0:
        print("Epoch {} Train loss {} Test loss = {}".format(epoch, train_loss, test_loss))
        
print("Training finished")


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg.coef_.reshape(n_dim, 1)

y_pred = lin_reg.predict(X_test)

y_model_pred = sess.run(pred, feed_dict={X : X_test, Y: y_test})

from sklearn.metrics import mean_squared_error

def rmse_evaluate(y_pred, y_test):
    lin_mse = mean_squared_error(y_pred, y_test)
    return np.sqrt(lin_mse)

print(rmse_evaluate(y_pred, y_test))
print(rmse_evaluate(y_model_pred, y_test))