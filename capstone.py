#Load the packages needed
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import display 
from IPython.display import Latex

#Split the data into training and testing set using load_data provided by the lstm package
nasdaq = pd.read_csv("Nasdaq.csv")

feature_cols = list(nasdaq.columns[:-1])
target_col = nasdaq.columns[-1]
X = nasdaq[feature_cols]
y = nasdaq[target_col]
print("Feature columns:\n{}".format(feature_cols))
print("\nTarget column: {}".format(target_col))
print("\nFeature values:")
print(X.head())

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
print(X.head())



X_tr, X_test,y_tr, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
X_train,  X_valid, y_train, y_valid = train_test_split(X_tr,y_tr, test_size = 0.2, random_state = 42)

#Step 2 Build Model
clf_1 = LinearRegression()
clf_2 = RandomForestRegressor()
clf_3 = SVR(kernel = 'rbf')

#Fit training data to models
clf_1.fit(X_train,y_train)
clf_2.fit(X_train,y_train)
clf_3.fit(X_train,y_train)

#Get Predictions from each model on the training data
predicted_1 = clf_1.predict(X_train)
predicted_2 = clf_2.predict(X_train)
predicted_3 = clf_3.predict(X_train)

#Get the mean squared error of the training set with each model
print(mean_squared_error(predicted_1,y_train))
print(mean_squared_error(predicted_2,y_train))
print(mean_squared_error(predicted_3,y_train))

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

#  Create the parameters list you wish to tune
parameters = {'n_estimators':[100,200,300], 'max_depth': [25,50,100,200,300]}

# TODO: Initialize the classifier
clf = RandomForestRegressor(random_state=42)

ms_scorer = make_scorer(mean_squared_error)
grid_obj = GridSearchCV(clf,parameters, scoring= ms_scorer)
grid_obj = grid_obj.fit(X_valid,y_valid)

# Get the estimator
clf = grid_obj.best_estimator_

#print(clf)
pred = clf.predict(X_valid)
print(mean_squared_error(pred,y_valid))
#Training set
predTest = clf.predict(X_test)
print(mean_squared_error(predTest, y_test))
### Part 2
import os
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
import lstm

X_train,y_train, X_test, y_test = lstm.load_data("NasdaqC.csv", 50, True)
print(X_train.shape)

# Build Siraj's modified LSTM model
model = Sequential()
model.add(LSTM(input_dim = 1, output_dim = 50, return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.5))
model.add(Activation("relu"))
model.add(Dense(output_dim = 1))
model.summary()

model2 = Sequential()
model2.add(Dense(output_dim = 1,input_shape = (50,1)))
model2.add(LSTM(100))
model2.add(Dropout(0.5))
model2.add(Dense(output_dim = 1))
model2.summary()

model3 = Sequential()
model3.add(Dense(output_dim = 1,input_shape = (50,1)))
model3.add(LSTM(100))
model3.add(Dropout(0.5))
model3.add(Dense(output_dim = 50))
model3.add(Dropout(0.4))
model3.add(Dense(output_dim = 25))
model3.add(Dropout(0.3))
model3.add(Dense(output_dim = 10))
model3.add(Dropout(0.2))
model3.add(Dense(output_dim = 5))
model3.add(Dense(output_dim = 1))
model3.summary()

start = time.time()
model.compile(loss = 'mse', optimizer = 'rmsprop')
print("compilation time: ", time.time() - start)

start = time.time()
model2.compile(loss = 'mse', optimizer = 'rmsprop')
print("compilation time: ", time.time() - start)
start = time.time()
model3.compile(loss = 'mse', optimizer = 'rmsprop')
print("compilation time: ", time.time() - start)

model.fit(X_train, y_train, batch_size = 512, nb_epoch = 7, validation_split = 0.05)

model2.fit(X_train, y_train, batch_size = 512, nb_epoch = 7, validation_split = 0.05)

model3.fit(X_train, y_train, batch_size = 512, nb_epoch = 7, validation_split = 0.05)

score1 = model.evaluate(X_test, y_test)
print("\n Mean Squared Error: ", score1)
score2 = model2.evaluate(X_test, y_test)
print("\n Mean Squared Error: ",score2)
score3 = model3.evaluate(X_test, y_test)
print("\n Mean Squared Error: ",score3)

predictions1 = lstm.predict_sequences_multiple(model,X_test, 50,50)
lstm.plot_results_multiple(predictions1,y_test, 50)

predictions2 = lstm.predict_sequences_multiple(model2,X_test, 50,50)
lstm.plot_results_multiple(predictions2,y_test, 50)

predictions3 = lstm.predict_sequences_multiple(model3,X_test, 50,50)
lstm.plot_results_multiple(predictions3,y_test, 50)