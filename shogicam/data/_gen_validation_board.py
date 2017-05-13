import cv2
import glob
import numpy as np
from pathlib import Path
from shogicam.constant import *
import shogicam.util

def gen_validation_board(img_dir, outdata_path):
    imgs = sorted(glob.glob(img_dir + '/*'))
    boards = []
    for path in imgs:
        raw_img = shogicam.util.load_img(path)
        corners, score = shogicam.preprocess.detect_corners(raw_img)
        board = shogicam.preprocess.trim_board(raw_img, corners)
        boards.append(board)

    board_cells = []
    for img in boards:
        dx = img.shape[0] / 9
        dy = img.shape[1] / 9
        def it():
            for i in range(9):
                for j in range(9):
                    sx = int(dx * i)
                    sy = int(dy * j)
                    cropped = img[sx:(int(sx + dx)), sy:(int(sy + dy))]
                    yield shogicam.util.normalize_img(cropped, IMG_ROWS, IMG_COLS)
        cells = np.array(list(it()))
        cells = cells.reshape(cells.shape[0], IMG_ROWS, IMG_COLS, 1)
        cells = cells.astype(np.float32)
        cells /= 255
        board_cells.append(cells)

    board_cells = np.array(board_cells)
    np.save(outdata_path, board_cells)
