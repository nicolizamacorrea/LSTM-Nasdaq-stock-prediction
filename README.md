# LSTM Nasdaq Stock prediction

This project involves testing out standard machine learning algorithms and long short term memory networks to predict nasdaq prices.

# Installation
You will need python 3.6 and the following python packages installed. Most of these come with anaconda, except for sklearn.

1. sklearn
2. time
3. pandas
4. numpy
5. IPython
Furthermore you will also need helper code lstm.py, which is taken from http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction

# Motivation

The motivation of this project is to compare the performance of lstm and "standard machine learning algorithms on the problem of predicting Nasdaq prices.
# Files

1. NasdaqO.csv: Original csv taken from yahoo
2. NasdaqC.csv: Modified original data for the lstm to use
3. lstm.py: helper code to preprocess data
4. capstone.py: code for training and testing machine learning algorithms
5. CapstoneReport.pdf: the report on this project

# Summary of the results
The main conclusion that we get from this project is that LSTMS are superior in performance when it comes to predicting prices, which bodes well with our hypothesis that LSTMs are specifically suited for time series analysis

# Acknowledgements
1. http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
