import cv2
import glob
import numpy as np
from pathlib import Path
from shogicam.constant import *
import shogicam.util

def gen_koma_traindata(img_dir, outdata_dir):
    saved_files = []
    for directory in sorted(glob.glob(img_dir + '/*')):
        found = []
        imgs = []
        if not Path(directory).is_dir():
            continue
        for label in LABELS:
            path = "%s/%s.png" % (directory, label)
            if Path(path).is_file():
                img = cv2.imread(path)
                img = shogicam.util.normalize_img(img, IMG_ROWS, IMG_COLS)
                imgs.append(img)
                found.append(label)
        if not imgs:
            continue
        file_path = '{0}/{1}.npz'.format(outdata_dir, Path(directory).stem)
        saved_files.append(file_path)
        np.savez_compressed(file_path, labels=found, imgs=imgs)
    return saved_files

def gen_empty_cell_traindata(img_dir, outdata_path):
    imgs = []
    saved_files = []
    for path in sorted(glob.glob(img_dir + '/*.png')):
        img = cv2.imread(path)
        img = shogicam.util.normalize_img(img, IMG_ROWS, IMG_COLS)
        imgs.append(img)
        saved_files.append(path)
    np.savez_compressed(outdata_path, imgs=imgs)
    return saved_files
