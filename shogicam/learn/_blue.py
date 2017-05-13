import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.models import load_model
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
    model = load_model("models/etl8.h5")
    for i in range(9):
        model.pop()
    for l in model.layers:
        l.trainable = False

    model.add(Dense(1024, name='shogi_dense_1'))
    model.add(BatchNormalization(name='shogi_norm_1'))
    model.add(PReLU(name='shogi_prelu_1'))
    model.add(Dropout(0.25, name='shogi_dropout_1'))
    model.add(Dense(1024, name='shogi_dense_2'))
    model.add(BatchNormalization(name='shogi_norm_2'))
    model.add(PReLU(name='shogi_prelu_2'))
    model.add(Dropout(0.5, name='shogi_dropout_2'))
    model.add(Dense(NUM_CLASSES, activation='softmax', name='shogi_dense_out'))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
    return model
