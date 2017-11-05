seed = 1
import numpy as np
import scipy.misc # to visualize only
import pandas as pd
import sklearn
import time

starttotal = time.time()
#dfmaster = pd.read_csv('train_x.csv', header = None)
dfmaster = pd.read_csv('/usr/local/pkgs/comp551_modified_mnist/train_x.csv', header = None)
labels = pd.read_csv('train_y.csv', header = None)


def cleaner(anydf):
    cutoff = 240
    for col in range(len(anydf.columns)):
        anydf.ix[anydf.loc[:,col]>=cutoff,col] = 255
        anydf.ix[anydf.loc[:,col]<cutoff,col] = 0
    return anydf

cleaner(dfmaster)

dfmaster['labels'] = labels.iloc[:,0]


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import initializers
from keras.layers import Conv2D,MaxPooling2D,Flatten, Dropout


noofcols = len(dfmaster.columns)

X_train, X_valid, y_train, y_valid = train_test_split(np.array(dfmaster.iloc[:,0:noofcols-1]/255), np.array(dfmaster.iloc[:,noofcols-1]) ,test_size=0.10, random_state = seed)



X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_valid = X_valid.reshape(X_valid.shape[0], 64, 64, 1)
#X_test /= 255


encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
dummy_y = np_utils.to_categorical(encoded_Y)
encodermapping = list(encoder.classes_)

model3 = Sequential()
model3.add(Conv2D(5, (3, 3),activation='relu',input_shape=(64,64,1)))
model3.add(Conv2D(5, (3, 3),activation='relu',input_shape=(64,64,1)))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(12, (3, 3),activation='relu'))
model2.add(Conv2D(12, (3, 3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(32, (3, 3),activation='relu'))
model2.add(Conv2D(32, (3, 3),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.4))
model2.add(Flatten())
model2.add(Dropout(0.4))
model2.add(Dense(200, activation='sigmoid'))
model2.add(Dense(len(encodermapping),activation='sigmoid', use_bias = True))

model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model2.fit(X_train, dummy_y, epochs=100, verbose = True, validation_split= 0.20)

model2.save('model170.h5')

predicttest = model2.predict(X_valid)



temp = []
for count in predicttest:
    temp.append(encodermapping[(np.argmax(count))])


print(100*sklearn.metrics.accuracy_score(y_valid,temp))



dftest = pd.read_csv('test_x.csv', header = None)
dftest = cleaner(dftest)
testx  = np.array(dftest.iloc[:,:]/255)
testx = testx.reshape(testx.shape[0], 64, 64, 1)

predicttest = model2.predict(testx)
temp = []
for count in predicttest:
    temp.append(encodermapping[(np.argmax(count))])

dftestout = pd.DataFrame(temp)

dftestout.to_csv("Submission.csv")
