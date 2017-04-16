from __future__ import print_function
import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import os

np.random.seed(11)

def shuffle(X, y):
    Xshape = X.shape
    xs = X.size
    xl = len(X)
    yshape = y.shape
    yl = len(y)
    c = np.c_[X.reshape(xl, -1), y.reshape(yl, -1)]
    np.random.shuffle(c)
    X = c[:, :xs/xl].reshape(Xshape)
    y = c[:, xs/xl:].reshape(yshape)
    return X, y

name = raw_input('enter name to save model: ')

X = np.load('temp/X.npy')
y = np.load('temp/y.npy')

X, y = shuffle(X, y)

X = X.astype('float32', copy=False)
X /= 255

X_train, X_test, y_train, y_test = train_test_split(X, y)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

batch_size = 32
num_classes = 8
epochs = 50

rms = RMSprop(lr=.00001)

model = Sequential()

model.add(Conv2D(36, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(36, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(72, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(72, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=rms,
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, Y_test),
          shuffle=True)

score = model.evaluate(X_test, Y_test, verbose=0)
print ('Test accuracy:', score[1])

while True:
    try:
        model.save('models/'+name+'.h5', 'wb')
    except Exception as ex:
        print (ex)
        raw_input()
        continue
    break

os.system('aws s3 cp models/'+name+'.h5 s3://python-objects/')
