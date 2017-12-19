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
from skimage.draw import line_aa
import random
import numpy as np

def data_generator(x_train, y_train, batch_size=16):
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=[0.55, 1.3],
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode="constant",
        cval=0.9
    )
    base = [(0, 0), (0, IMG_COLS - 1), (IMG_ROWS - 1, IMG_COLS - 1), (IMG_ROWS - 1, 0)]
    color = random.choice([True, False])

    for xs, ys in datagen.flow(x_train, y_train, batch_size=batch_size):
        line_cnt = random.randrange(5)
        for i in range(line_cnt):
            cs = np.random.randint(14, size=4)
            cs -= 7
            j = random.randrange(4)
            cs[0] += base[j][0]
            cs[1] += base[j][1]
            cs[2] += base[(j + 1) % 4][0]
            cs[3] += base[(j + 1) % 4][1]
            rr, cc, v = line_aa(cs[0], cs[1], cs[2], cs[3])
            idx = np.where((rr >= 0) & (rr < IMG_ROWS) & (cc >= 0) & (cc < IMG_ROWS))

            if color:
                xs[:, rr[idx], cc[idx], 0] = v[idx]
            else:
                xs[:, rr[idx], cc[idx], 0] = v[idx] * (-1) + 1.0
        yield (xs, ys)

def learn(data_dir, verbose=False, test_size=0.05):
    x_train, x_test, y_train, y_test = load_traindata_with_validation_board(data_dir, test_size)

    model = gen_model()
    if verbose:
        model.summary()

    model.fit_generator(data_generator(x_train, y_train, batch_size=16),
            steps_per_epoch=x_train.shape[0], epochs=120, verbose=verbose,
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
