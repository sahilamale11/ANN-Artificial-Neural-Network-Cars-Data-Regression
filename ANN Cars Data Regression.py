# -*- coding: utf-8 -*-
# deep learning - Regression
# dataset: cars

##################################################################
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# specific to neural networks
from keras.models import Sequential
from keras.layers import Dense
###################################################################

###################################################################
# read the file
path = "C:\\Users\\Sahil\\Desktop\\Imarticus Learning -4\\DAY 63\\cars.csv"
cars = pd.read_csv(path)
cars.head(10)
###################################################################

###################################################################
# remove feature 'name'
cars.drop(columns='name',inplace=True)
cars.head(10)

# include the EDA code here
###################################################################

###################################################################
# scale the dataset
cars_scaled = cars.copy()
minmax = preprocessing.MinMaxScaler()
cars_scaled.iloc[:,:] = minmax.fit_transform(cars_scaled.iloc[:,:])
cars_scaled.head(10)

# replace the Y with the original Y
cars_scaled.mpg = cars.mpg

# check both the datasets
cars.head(10)
cars_scaled.head(10)

len(cars_scaled)
####################################################################

####################################################################
# split the data into train and test
trainx,testx,trainy,testy = train_test_split(cars_scaled.drop('mpg',1),
                                             cars_scaled.mpg,
                                             test_size=0.25)
trainx.shape,trainy.shape
testx.shape, testy.shape
#####################################################################



#####################################################################
# ----------------------------------------------------------------------- 

# build the ANN 

def buildModel(units,shape,lr=0.001):
    model = keras.Sequential([
                Dense(units,activation='relu',input_shape=[shape]),
                Dense(units,activation='relu'),
                Dense(1) ])
    
    # optimise the learning rate    
    optimizer = tf.keras.optimizers.RMSprop(lr)
    
    # compile the model
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    
    return(model)

# build the model
units=10 # try diff combinations (16,32... any number)
shape = len(trainx.columns)

m1 = buildModel(units,shape)

# summarise the model
m1.summary()

# train the model
EPOCHS = 1000
e1 = m1.fit(trainx,trainy,epochs=EPOCHS,validation_split=0.2)

# difference between training and validation errors
hist_err = pd.DataFrame(e1.history)
print(hist_err)
hist_err['epoch'] = e1.epoch

# rearrange the columns
cols=['mae','mse','val_mae','val_mse','epoch']
hist_err = hist_err[cols]
hist_err.head(20)

# plot the errors
# plt.subplot(121)
plt.plot(hist_err.epoch,hist_err.mse,label='Val MSE')
plt.title("MSE")

# prediction on test data
p1 = m1.predict(testx).flatten() # transform 2-D to 1-D
p1 = np.round(p1.astype(float),1)

# dataframe to store actual and predicted data
df1 = pd.DataFrame({'actual':testy, 'predicted':p1})

# MSE
mse1 = mean_squared_error(testy,p1)

print('Model Error\n\tMSE = {},\n\tRMSE = {}'.format(mse1,np.sqrt(mse1)))
########################################################################



########################################################################
# how to optimise the model performance
# i) do the EDA (esp check for multicollinearity)
# ii) change the units (nodes)
# iii) change EPOCHS
# iv) change the LR
# v) change the levels of network
# vi) select only the best features
########################################################################

