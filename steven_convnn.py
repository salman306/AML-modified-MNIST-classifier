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

print "> Reading training sets"
X = pd.read_csv("processed_train_x.csv")
X = X.iloc[:,1:]
Y = pd.read_csv("train_y.csv", header = None)

encoder = LabelEncoder()
encoder.fit(Y) #transform: encode to label, inverse: get back class
encodermapping = list(encoder.classes_)

seed = 8
X_train, X_valid, y_train, y_valid = train_test_split(X.as_matrix()/255, Y.as_matrix(), test_size=0.1, random_state = seed)

X_train = X_train.reshape(X_train.shape[0],64,64,1)
X_valid = X_valid.reshape(X_valid.shape[0], 64,64,1)

y_train = np_utils.to_categorical(encoder.transform(y_train))
y_valid= np_utils.to_categorical(encoder.transform(y_valid))

print "X_train:", X_train.shape, y_train.shape
print "X_valid:", X_valid.shape, y_valid.shape

batch_size = 128
epochs = 100


model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(64,64,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(128, (5,5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (5,5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=1))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(80))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dense(len(encoder.classes_)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(X_valid, y_valid))

predict_validation = model.predict(X_valid)

temp = []
for instance in predict_validation:
	temp.append(encodermapping[np.argmax(instance)])
		    
print(100*sklearn.metrics.accuracy_score(y_valid, temp))

print("> Reading testing set")
X_test = pd.read_csv("processed_test_x.csv")
X_test = X_test.iloc[:,1:]

X_test = X_test.as_matrix()/255
X_test = X_test.reshape(X_test.shape[0], 64,64,1)

predict_test = model.predict(X_test)
temp_test = []
for instance in predict_test:
	temp_test.append(encodermapping[np.argmax(nstance)])

dftestout = pd.DataFrame(temp_test)
dftestout.to_csv("submission.csv")



