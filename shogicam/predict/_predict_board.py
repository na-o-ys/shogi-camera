import numpy as np
from keras import backend as K
from keras.models import load_model
from shogicam.constant import *
import shogicam.util

def predict_board(img, model_path):
    cell_imgs = cells(img)
    model = load_model(model_path)
    y = model.predict(cell_imgs)
    K.clear_session()
    return np.array([np.argmax(c) for c in y])

def cells(img):
    dx = img.shape[0] / 9
    dy = img.shape[1] / 9
    def it():
        for i in range(9):
            for j in range(9):
                sx = int(dx * i)
                sy = int(dy * j)
                cropped = img[sx:(int(sx + dx)), sy:(int(sy + dy))]
                yield shogicam.util.normalize_img(cropped, IMG_ROWS, IMG_COLS)
    cs = np.array(list(it()))
    cs = cs.reshape(cs.shape[0], IMG_ROWS, IMG_COLS, 1)
    cs = cs.astype(np.float32)
    cs /= 255
    return cs
