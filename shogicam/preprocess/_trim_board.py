import cv2
import numpy as np
from ._detect_corners import detect_corners

BASE_SIZE = 64
def trim_board(img, corners):
    w = BASE_SIZE * 14
    h = BASE_SIZE * 15
    transform = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    normed = cv2.warpPerspective(img, transform, (w, h))
    return normed
