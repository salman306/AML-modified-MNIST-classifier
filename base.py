import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
from keras.utils import np_utils

BATCH_SIZE = 128
EPOCHS = 60
VALIDATION = False
SEED = 8

print "> Reading training sets"

X_train = pd.read_csv("processed_train_x.csv")
X_train = X.iloc[:,1:]
y_train = pd.read_csv("train_y.csv", header = None)

# setup one hot encoder
encoder = LabelEncoder()
encoder.fit(y_train) #transform: encode to label, inverse: get back class
encodermapping = list(encoder.classes_)

#GET MODEL
model = NNConvModels('model_salman')

if VALIDATION:
	X_train, X_valid, y_train, y_valid = train_test_split(X_train.as_matrix()/255, y_train.as_matrix(), test_size=0.1, random_state = seed)

	X_valid = X_valid.reshape(X_valid.shape[0], 64,64,1)
	X_train = X_train.reshape(X_train.shape[0], 64,64,1)

	y_train = np_utils.to_categorical(encoder.transform(y_train))
	y_valid_hot = np_utils.to_categorical(encoder.transform(y_valid))

	print("> Train/Validation split")
	print("> 	X_train:", X_train.shape, y_train.shape)
	print(">	X_valid:", X_valid.shape, y_valid.shape)
 
 	model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, validation_data=(X_valid, y_valid_hot))

	print("> Reading testing set")
	X_test = pd.read_csv("processed_test_x.csv")
	X_test = X_test.iloc[:,1:]

	X_test = X_test.as_matrix()/255
	X_test = X_test.reshape(X_test.shape[0], 64,64,1)

	predict_test = model.predict(X_test)
	temp_test = []
	for instance in predict_test:
		        temp_test.append(encodermapping[np.argmax(instance)])

	dftestout = pd.DataFrame(temp_test)
	dftestout.to_csv("submission.csv")
 
else:
	X_train = X_train.reshape(X_train.shape[0], 64,64,1)
	y_train = np_utils.to_categorical(encoder.transform(y_train))

 	model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)

	print("> Reading testing set")

	X_test = pd.read_csv("processed_test_x.csv")
	X_test = X_test.iloc[:,1:]

	X_test = X_test.as_matrix()/255
	X_test = X_test.reshape(X_test.shape[0], 64,64,1)

	predict_test = model.predict(X_test)
	temp_test = []
	for instance in predict_test:
		        temp_test.append(encodermapping[np.argmax(instance)])

	dftestout = pd.DataFrame(temp_test)
	dftestout.to_csv("submission.csv")

print("> Done")


#Validation or not

# FIT TRAIN



