import cv2
import glob
import numpy as np
import struct
from PIL import Image
from scipy.misc import imresize
from shogicam.constant import *
import shogicam.util

RECORDS_PER_DATASET = 956

def gen_etl8(etl8_dir, outdata_path):
    data = read_etl8(etl8_dir)
    np.savez_compressed(outdata_path, data=data)

def read_record_img(f):
    s = f.read(8199)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    pil = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    pil = pil.convert('L')
    img = np.array(pil, np.uint8) * 16
    return imresize(img, (IMG_ROWS, IMG_COLS), interp='bilinear')

def read_dataset(f):
    ret = np.zeros((RECORDS_PER_DATASET, IMG_ROWS, IMG_COLS))
    for i in range(RECORDS_PER_DATASET):
        ret[i] = read_record_img(f)
    return ret

def read_etl8_file(fname):
    ret = np.zeros((5, RECORDS_PER_DATASET, IMG_ROWS, IMG_COLS))
    with open(fname, 'rb') as f:
        for i in range(5):
            f.seek(i * RECORDS_PER_DATASET * 8199)
            ret[i] = read_dataset(f)
    return ret

def read_etl8(etl8_dir):
    ret = np.zeros((160, RECORDS_PER_DATASET, IMG_ROWS, IMG_COLS))
    for i in range(32):
        fname = etl8_dir + ("/ETL8G_%02d" % (i + 1))
        ret[(i * 5):(i * 5 + 5)] = read_etl8_file(fname)
    return ret
