import cv2
import numpy as np
from shogicam.constant import *

def load_img(img_path):
    return cv2.imread(img_path)

def draw_rect(img, rect):
    cntr = np.int32(rect.reshape((4, 2)))
    blank = np.copy(img)
    cv2.drawContours(blank, [cntr], -1, (0,255,0), 2)
    return blank

def save(img, path):
    cv2.imwrite(path, img)

def normalize_img(img, h, w):
    size = img.shape[:2]
    f = min(h / size[0], w / size[1])
    resized = cv2.resize(img, (int(size[1] * f), int(size[0] * f)), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blank = np.full((h, w), np.uint8(255), dtype=np.uint8)
    hstart = int((h - gray.shape[0]) / 2)
    wstart = int((w - gray.shape[1]) / 2)
    blank[hstart:(hstart + gray.shape[0]), wstart:(wstart + gray.shape[1])] = gray
    return blank

def label_name(idx):
    if len(LABELS) > idx:
        return " " + LABELS_JA[idx]
    elif len(LABELS) * 2 > idx:
        return "v" + LABELS_JA[idx - len(LABELS)]
    else:
        return " ・"

def boardfile_to_content(f):
    ret = np.empty((9, 9))
    for row, line in enumerate(f):
        for i in range(9):
            koma = line[3 * i + 1]
            if koma in LABELS_JA:
                idx = LABELS_JA.index(koma)
            else:
                idx = len(LABELS_JA) * 2
            if line[3 * i] == 'v':
                idx += len(LABELS_JA)
            ret[row, i] = idx
    return ret
