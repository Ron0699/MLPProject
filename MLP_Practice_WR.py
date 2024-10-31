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

End = datetime.datetime(year=2009, month=12, day=1)
Start = datetime.datetime(year=2007, month=1, day=1)

#df = web.DataReader('^DJI', 'yahoo', Start, End)
df = web.DataReader('AAPL', 'yahoo', Start, End)
plt.plot(df.index, df.Close, 'gray')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)

#Target - y
DaysPeriod = 20
WILLR = talib.WILLR(df['High'], df['Low'], df['Close'], DaysPeriod)
DayWILLR = np.where( (WILLR < -95) &
                    (df.Close.shift(-1) >= df.Close) &
                    (df.Close.shift(-2) >= df.Close.shift(-1))
                    , 1, 0)
for i in range(len(df)):
    if DayWILLR[i] == 1:
        plt.plot(df.index[i], df.Close[i], '*r')
#plt.savefig('DL.jpg', dpi=200)
#plt.close()

#Training samples - Factor X
train_X = df.iloc[:,:4].copy()
#print(train_X.index)
#print(train_X.columns)
#print(train_X.iloc[0, 1])
for i in range(1, DaysPeriod):
    train_X = pd.concat([train_X, df.Close.shift(i)], axis = 1)
train_y = DayWILLR[:].copy()

#Delete Nan Values
train_X = train_X[DaysPeriod:]
train_y = train_y[DaysPeriod:]

#Normalize
minmax = preprocessing.MinMaxScaler()
train_X = minmax.fit_transform(train_X)

clf = MLPClassifier(hidden_layer_sizes=(180, 180, 180, 180, 180, 180, 180, 180),
                                                #neuron values in each hidden layer
                    activation='relu',          #identity, logistic, tanh, relu
                    solver='lbfgs',             #lbfgs, adam, sgd
                    alpha=1e-5,
                    batch_size='auto',
                    learning_rate='adaptive',   #constant, invscaling, adaptive
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=1600,               #tol or max_iter
                    shuffle=True,               #random or not when every time iter
                    random_state=None,
                    tol=0.0001,
                    verbose=0,                  #0:no output, 1:occasionally, 2:always
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-08)

#Training
clf.fit(train_X, train_y)
#print(clf.fit(train_X, train_y))

#Save the Model
filename = 'MLP_WR.sav'
pickle.dump(clf, open(filename, 'wb'))

#Print the simple result
print(clf.n_layers_)
print(clf.n_iter_)
print(clf.loss_)
print(clf.out_activation_)