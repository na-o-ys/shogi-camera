import numpy as np
from keras import backend as K
from keras.models import load_model
from shogicam.constant import *
import shogicam.util

def eval_model(model_path, x, y):
    model = load_model(model_path)
    loss, acc = model.evaluate(x, y, verbose=False)
    print("acc: {} loss: {}".format(acc, loss))
    K.clear_session()
