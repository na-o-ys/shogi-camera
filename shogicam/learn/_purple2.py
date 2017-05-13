import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from ._load_traindata import load_traindata_nosplit
from shogicam.constant import *
import shogicam.data

def learn(data_dir, verbose=False, test_size=0.05):
    x_train, y_train = load_traindata_nosplit(data_dir, test_size)
    x_test, y_test = shogicam.data.load_validation_board_data(data_dir + '/board')
    model = gen_model()
    if verbose:
        model.summary()

    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=[0.55, 1.0],
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode="constant",
        cval=0.9
    )
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
            steps_per_epoch=x_train.shape[0], epochs=70, verbose=verbose,
            validation_data=(x_test, y_test))
    return model

def gen_model():
    input_shape = (IMG_ROWS, IMG_COLS, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.25))
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
    return model
