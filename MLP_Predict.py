import requests, datetime, xlrd, os, random, time, talib, warnings
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy import signal
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import pickle

#Load the Model
filename = 'MLP_WR.sav'
model = pickle.load(open(filename, 'rb'))

End = datetime.date.today()
End = datetime.datetime(year=2022 , month=9, day=30)
Start = datetime.datetime(year=2020 , month=11, day=1)
#df = web.DataReader('^DJI', 'yahoo', Start, End)
df = web.DataReader('AAPL', 'yahoo', Start, End)
plt.plot(df.index, df.Close, 'gray')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)

#Test samples
DaysPeriod = 20
test_X = df.iloc[:,:4].copy()
for i in range(1, DaysPeriod):
    test_X = pd.concat([test_X, df.Close.shift(i)], axis = 1)
test_X = test_X[DaysPeriod:]
#Normalize
minmax = preprocessing.MinMaxScaler()
test_X = minmax.fit_transform(test_X)

#Start Predict
PredictResult = model.predict(test_X)
#print(PredictResult)

#Plot Result
for i in range(0, len(df)-DaysPeriod):
    if PredictResult[i] == 1:
        plt.plot(df.index[DaysPeriod+i], df.Close[DaysPeriod+i], '*r')
#plt.savefig('DL.jpg', dpi=200)
#plt.close()

#Plot WR
WILLR = talib.WILLR(df['High'], df['Low'], df['Close'], DaysPeriod)
for i in range(0, len(df)):
    if WILLR[i] < -95:
        plt.plot(df.index[i], df.Close[i], '+b')