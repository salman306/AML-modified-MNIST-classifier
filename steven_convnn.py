import keras
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X.as_matrix()/255, Y.as_matrix(), test_size=0.1, random_state = seed)

X_train = X_train.reshape(X_train.shape[0],64,64,1)
X_test = X_test.reshape(X_test.shape[0], 64,64,1)

y_train = np_utils.to_categorical(encoder.transform(y_train))

print "X_train:", X_train.shape, y_train.shape
print "X_test:", X_test.shape, y_test.shape


model = Sequential()
model.add(Conv2D(6, (5,5), input_shape=(64,64,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
model.add(Conv2D(16, (5,5)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(len(encoder.classes_)))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=True, validation_split=0.20)

model.save("steven_model.h5")
predict_validation = model.predict(X_test)

temp = []
for instance in predict_validation:
	temp.append(encodermapping[np.argmax(instance)])
		    
print(100*sklearn.metrics.accuracy_score(y_test, temp))



