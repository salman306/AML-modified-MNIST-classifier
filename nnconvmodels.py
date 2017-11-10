import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers


class NNConvModels():
	""" Functions hold the convolutional models """


	def __init__(self, model_name):
		print "> Retrieving the model:", model_name
		self.NUM_CLASSES = 40
		self.model_name = model_name

	def getModel(self):
		return getattr(self, self.model_name)()

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
		model.add(Dense(self.NUM_CLASSES,activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model

	def model_7(self):
		model = Sequential()

		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(200, activation='tanh'))
		model.add(Dense(self.NUM_CLASSES,activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model


	def model_8(self):
		model = Sequential()

		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(300, activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(300, activation='tanh'))
		model.add(Dense(self.NUM_CLASSES,activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model



	def model_9(self):
		model = Sequential()

		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(Conv2D(30, (3, 3),activation='relu',input_shape=(64,64,1)))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(Conv2D(60, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(Conv2D(120, (2, 2),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(300, activation='tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(300, activation='tanh'))
		model.add(Dense(self.NUM_CLASSES,activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model


	def model_top(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(1000, activation='tanh'))
                model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top1(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(1000, activation='tanh'))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top2(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(1000, activation='tanh'))
		model.add(Dropout(0.2))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top3(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(1000, activation='tanh'))
		model.add(Dropout(0.5))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top4(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top5(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.2))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top6(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(96, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.5))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top7(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top8(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
		model.add(Dropout(0.5))
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top9(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.9))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top10(self):
		""" best model so far 93% """
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
                model.add(Flatten())
		model.add(Dropout(0.5))
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top11(self):
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
		model.add(Dropout(0.5))
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top12(self):
		"""93.90 (48 epoch)"""
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top13(self):
		"""94.04 (47 epochs)"""
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(30, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top14(self):
		"""94.14 (42 epochs)"""
		model = Sequential()
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(20, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top15(self):
		"""94.24 (58 epochs)"""
		model = Sequential()
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top16(self):
		"""94.28 (30 epochs)"""
		model = Sequential()
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(42, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(42, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(96, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top17(self):
		"""? (30 epochs)"""
		model = Sequential()
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (5, 5),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(42, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(42, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(104, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(104, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top18(self):
		"""94.4 (60 epochs)"""
		model = Sequential()
                model.add(Conv2D(64, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(64, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(94, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(94, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(118, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(118, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top19(self):
		"""90 (30 epochs)"""
		model = Sequential()
                model.add(Conv2D(28, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(90, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(90, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(50, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(50, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top20(self):
		"""80 (30 epochs)"""
		model = Sequential()
                model.add(Conv2D(28, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(44, (3, 3),activation='relu', padding='same', strides=2))
                model.add(Conv2D(44, (3, 3),activation='relu', padding='same', strides=2))
                model.add(MaxPooling2D(pool_size=(2,2)))
                model.add(Conv2D(92, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(92, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(2000, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model


	def model_steven_crazy(self):
		""" 92.2274 """
		model = Sequential()
		model.add(Conv2D(20, (3, 3),input_shape=(64,64,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(20, (3, 3),input_shape=(64,64,1), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(30, (3, 3), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(30, (3, 3), padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Conv2D(80, (3, 3),padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Conv2D(80, (3, 3),padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Flatten())
		model.add(Dense(1024, activation='tanh'))
		model.add(Dropout(0.59))
		model.add(Dense(512, activation='tanh'))
		model.add(Dropout(0.25))
		model.add(Dense(256, activation='tanh'))
		model.add(Dropout(0.25))
		model.add(Dense(self.NUM_CLASSES,activation='sigmoid', use_bias = True))

		model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		return model

	def model_top21(self):
		"""94.4 (60 epochs)"""
		model = Sequential()
                model.add(Conv2D(256, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(256, (3, 3),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(80, (3, 3),activation='relu', padding='same'))
                model.add(Conv2D(80, (3, 3),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(Conv2D(36, (5, 5),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top22(self):

		model = Sequential()
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(96, (18, 18),activation='relu', padding='same'))
                model.add(Conv2D(96, (18, 18),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top23(self):

		model = Sequential()
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(128, (18, 18),activation='relu', padding='same'))
                model.add(Conv2D(128, (18, 18),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top24(self):

		model = Sequential()
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(28, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(Conv2D(36, (36, 36),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(128, (18, 18),activation='relu', padding='same'))
                model.add(Conv2D(128, (18, 18),activation='relu', padding='same'))
                model.add(Conv2D(128, (18, 18),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model

	def model_top25(self):

		model = Sequential()
                model.add(Conv2D(64, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(Conv2D(64, (72, 72),activation='relu',input_shape=(64,64,1), padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(128, (36, 36),activation='relu', padding='same'))
                model.add(Conv2D(128, (36, 36),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Conv2D(256, (18, 18),activation='relu', padding='same'))
                model.add(Conv2D(256, (18, 18),activation='relu', padding='same'))
                model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(200, activation='tanh'))
		model.add(Dropout(0.75))
                model.add(Dense(200, activation='tanh'))
                model.add(Dense(self.NUM_CLASSES,activation='softmax', use_bias = True))

                model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
		
		return model
