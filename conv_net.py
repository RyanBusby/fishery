from __future__ import print_function
import numpy as np
from PIL import Image
from keras.optimizers import Adagrad, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import backend as K
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import os

np.random.seed(11)

# os.system('aws s3 cp s3://rb-fishery-python-objects/X.npy temp/')
# os.system('aws s3 cp s3://rb-fishery-python-objects/labels.npy temp/')

X = np.load('temp/test_prep_X.npy')
y = np.load('temp/test_prep_y.npy')

print (X.shape)
img_rows, img_cols = X.shape[1], X.shape[2]
batch_size = 200
n_f = 64
nb_classes = 8
nb_epoch = 25

X = X.astype('float32', copy=False)
X /= 255

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1,  img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
print (input_shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(n_f, 3, 3,
                    border_mode='valid',
                    input_shape=input_shape))

model.add(Activation('relu'))
model.add(Convolution2D(n_f, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(n_f*2, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(n_f*2, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1], ' this is the one we care about.')


while True:
    try:
        model.save('models/prep_model.h5', 'wb')
    except Exception as ex:
        print (ex)
        raw_input()
        continue
    break

os.system('aws s3 cp models/prep_model64filters.h5 s3://python-objects/')
