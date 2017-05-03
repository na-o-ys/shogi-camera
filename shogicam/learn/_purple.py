import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from ._load_traindata import load_traindata
from shogicam.constant import *

def learn(data_dir, verbose=False, test_size=0.05):
    x_train, x_test, y_train, y_test = load_traindata(data_dir, test_size)
    model = gen_model()
    if verbose:
        model.summary()

    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.25)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
            steps_per_epoch=x_train.shape[0], epochs=80, verbose=verbose,
            validation_data=(x_test, y_test))
    return model

def gen_model():
    input_shape = (IMG_ROWS, IMG_COLS, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
    return model
