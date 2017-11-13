import keras
import sys
import pandas as pd
import numpy as np
from nnconvmodels import NNConvModels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard

BATCH_SIZE = 256
EPOCHS = 31
VALIDATION = False
SEED = 8

print "> Reading training sets"

X_train = pd.read_csv("train_x.csv", header = None)
y_train = pd.read_csv("train_y.csv", header = None)
#X_train = pd.read_csv("processed_train_x.csv", header = None)
#y_train = pd.read_csv("train_y.csv", header = None)
#X_train = pd.read_csv("concat_train_x.csv", header = None)
#y_train = pd.read_csv("concat_train_y.csv", header = None)

# setup one hot encoder
encoder = LabelEncoder()
encoder.fit(y_train) #transform: encode to label, inverse: get back class
encodermapping = list(encoder.classes_)

#GET MODEL
model_name = sys.argv[1]
model = NNConvModels(model_name).getModel()

if VALIDATION:
	X_train, X_valid, y_train, y_valid = train_test_split(X_train.as_matrix()/255.0, y_train.as_matrix(), test_size=0.1, random_state = SEED)

	X_valid = X_valid.reshape(X_valid.shape[0], 64,64,1)
	X_train = X_train.reshape(X_train.shape[0], 64,64,1)

	y_train = np_utils.to_categorical(encoder.transform(y_train))
	y_valid_hot = np_utils.to_categorical(encoder.transform(y_valid))

	print("> Train/Validation split")
	print("> 	X_train:", X_train.shape, y_train.shape)
	print(">	X_valid:", X_valid.shape, y_valid.shape)

	filepath = str(model_name)+"_raw_bestweights.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor="val_acc",save_best_only=True, mode='max')
	tbCallback = TensorBoard(log_dir='./Graph_'+str(model_name)+"_raw", histogram_freq=0, write_graph=True, write_images=True)
	earlyStop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0002, patience=20, verbose=0, mode='auto')
	callback_lists = [checkpoint, earlyStop, tbCallback]
 
	history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, callbacks=callback_lists, validation_data=(X_valid, y_valid_hot))


	print("> Reading testing set")
	#X_test = pd.read_csv("processed_test_x.csv", header = None)
	X_test = pd.read_csv("test_x.csv", header = None)

	X_test = X_test.as_matrix()/255
	X_test = X_test.reshape(X_test.shape[0], 64,64,1)

	predict_test = model.predict(X_test)
	temp_test = []
	for instance in predict_test:
		temp_test.append(encodermapping[np.argmax(instance)])

	dftestout = pd.DataFrame(temp_test)
	dftestout.to_csv("submission.csv")
 
else:
	X_train = X_train.as_matrix()/255.0
	X_train = X_train.reshape(X_train.shape[0], 64,64,1)
	y_train = np_utils.to_categorical(encoder.transform(y_train))

 	#model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)
	filepath = str(model_name)+"_raw_bestweights.hdf5"

	tbCallback = TensorBoard(log_dir='./Graph_'+str(model_name)+"_raw", histogram_freq=0, write_graph=True, write_images=True)
	callback_lists = [tbCallback]
 
	history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True, callbacks=callback_lists)

	#model.load_weights(filepath)
	#model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	print("> Reading testing set")

	#X_test = pd.read_csv("processed_test_x.csv", header = None)
	X_test = pd.read_csv("test_x.csv", header = None)

	X_test = X_test.as_matrix()/255
	X_test = X_test.reshape(X_test.shape[0], 64,64,1)

	predict_test = model.predict(X_test)
	temp_test = []
	for instance in predict_test:
		temp_test.append(encodermapping[np.argmax(instance)])

	dftestout = pd.DataFrame(temp_test)
	dftestout.to_csv("submission.csv")

	


print("> Done "+str(model_name))
