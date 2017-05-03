import numpy as np
import glob
from shogicam.constant import *

def load_data(data_dir):
    series_imgs, series_labels = load_koma_series(data_dir + "/koma")
    x_front, y_front = series_to_array(series_imgs, series_labels)
    x_rot, y_rot = rotate180(x_front, y_front)
    x_space, y_space = load_empty_cell(data_dir + "/empty_cell.npz")
    return np.r_[x_front, x_rot, x_space], np.r_[y_front, y_rot, y_space]

def load_koma_series(koma_dir):
    series_imgs = []
    series_labels = []
    for series in sorted(glob.glob(koma_dir + "/*.npz")):
        f = np.load(series)
        series_imgs.append(f['imgs'].astype(np.float32))
        series_labels.append(f['labels'])
    return series_imgs, series_labels

def series_to_array(series_imgs, series_labels):
    x = np.empty((0, IMG_ROWS, IMG_COLS))
    y = np.empty((0, 2), np.int32)
    for i in range(len(series_imgs)):
        num_imgs = len(series_imgs[i])
        x = np.r_[x, series_imgs[i]]
        label_indices = []
        for j, label in enumerate(series_labels[i]):
            label_indices.append(LABELS.index(label))
        label_and_series = np.c_[label_indices, np.full((num_imgs), i)]
        y = np.r_[y, label_and_series]
    return x, y

def rotate180(x, y):
    x_rot = np.rot90(x, 2, (1, 2))
    y_rot = np.copy(y)
    y_rot[:, 0] += len(LABELS)
    return x_rot, y_rot

def load_empty_cell(path):
    x = np.load(path)['imgs'].astype(np.float32)
    y = np.full((len(x), 2), [len(LABELS) * 2, 1000])
    return x, y
