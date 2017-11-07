import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers


class NNConvModels():
	""" Functions hold the convolutional models """

	def __init__(self, model_name):
		return getattr(self, model_name)()	
	
	def model_salman(self):
		""" Returns model """

		model = Sequential()

		model.add(Conv2D(20, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
		model.add(Conv2D(20, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(30, (3, 3),activation='relu'))
		model.add(Conv2D(30, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(80, (3, 3),activation='relu'))
		model.add(Conv2D(80, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(200, activation='tanh'))
		model.add(Dense(len(encodermapping),activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model

	def model_steven_samepad(self):
		model = Sequential()
		model.add(Conv2D(20, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
		model.add(Conv2D(20, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(30, (3, 3),activation='relu', padding='same'))
		model.add(Conv2D(30, (3, 3),activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
		model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(200, activation='tanh'))
		model.add(Dense(len(encodermapping),activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model
