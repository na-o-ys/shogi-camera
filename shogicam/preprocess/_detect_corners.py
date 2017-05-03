import cv2
import numpy as np
import itertools
import math
from scipy.optimize import basinhopping

def detect_corners(raw_img):
    img = fit_size(raw_img, 500, 500)
    polies = convex_poly(img)
    poly = select_corners(img, polies)
    x0 = poly.flatten()
    rect, score = convex_poly_fitted(img)

    scale = raw_img.shape[0] / img.shape[0]
    rect = np.int32(normalize_corners(rect) * scale)

    score *= -1/255
    return rect, score

def fit_size(img, h, w):
    size = img.shape[:2]
    f = min(h / size[0], w / size[1])
    return cv2.resize(img, (int(size[1] * f), int(size[0] * f)), interpolation=cv2.INTER_AREA)

def edge(img, show=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def line(img, show=True, threshold=80, minLineLength=50, maxLineGap=5):
    edges = edge(img, False)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, 200, minLineLength, maxLineGap)
    return lines

def contours(img, show=True):
    edges = edge(img, False)
    contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    blank = np.zeros(img.shape, np.uint8)
    min_area = img.shape[0] * img.shape[1] * 0.2 # 画像の何割占めるか
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    cv2.drawContours(blank, large_contours, -1, (0,255,0), 1)
    return large_contours

def convex(img, show=True):
    blank = np.copy(img)
    convexes = []
    for cnt in contours(img, False):
        convex = cv2.convexHull(cnt)
        cv2.drawContours(blank, [convex], -1, (0,255,0), 2)
        convexes.append(convex)
    return convexes

def convex_poly(img, show=True):
    cnts = convex(img, False)
    blank = np.copy(img)
    polies = []
    for cnt in cnts:
        arclen = cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, 0.02*arclen, True)
        cv2.drawContours(blank, [poly], -1, (0,255,0), 2)
        polies.append(poly)
    return [poly[:, 0, :] for poly in polies]

def select_corners(img, polies):
    p_selected = []
    p_scores = []
    for poly in polies:
        choices = np.array(list(itertools.combinations(poly, 4)))
        scores = []
        # 正方形に近いものを選ぶ
        for c in choices:
            line_lens = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
            base = cv2.contourArea(c) ** 0.5
            score = sum([abs(1 - l/base) ** 2 for l in line_lens])
            scores.append(score)
        idx = np.argmin(scores)
        p_selected.append(choices[idx])
        p_scores.append(scores[idx])
    return p_selected[np.argmin(p_scores)]

def gen_score_mat():
    half_a = np.fromfunction(lambda i, j: ((10 - i) ** 2) / 100.0, (10, 20), dtype=np.float32)
    half_b = np.rot90(half_a, 2)
    cell_a = np.r_[half_a, half_b]
    cell_b = np.rot90(cell_a)
    cell = np.maximum(cell_a, cell_b)
    return np.tile(cell, (9, 9))

SCALE = 0.7
def get_get_fit_score(img, x):
    # 入力リサイズ
    img = cv2.resize(img, (int(img.shape[1] * SCALE), int(img.shape[0] * SCALE)), interpolation=cv2.INTER_AREA)
    img_size = (img.shape[0] * img.shape[1]) ** 0.5
    x = np.int32(x * SCALE)

    # 線分化
    poly_length = cv2.arcLength(x, True)
    lines = line(img, False, int(poly_length / 12), int(poly_length / 200))
    line_mat = np.zeros(img.shape, np.uint8)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_mat, (x1, y1), (x2, y2), 255, 1)
    line_mat = line_mat[:, :, 0]
    
    # 矩形の外をマスクアウト
    img_size = (img.shape[0] * img.shape[1]) ** 0.5
    mask = np.zeros(line_mat.shape, np.uint8)
    cv2.fillConvexPoly(mask, x, 1)
    kernel = np.ones((int(img_size / 10), int(img_size / 10)), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    line_mat[np.where(mask == 0)] = 0
    
    # スコア
    score_mat = gen_score_mat()

    def get_fit_score(x):
        img_pnts = np.float32(x).reshape(4, 2)
        img_pnts *= SCALE
        score_size = score_mat.shape[0]
        score_pnts = np.float32([[0, 0], [0, score_size], [score_size, score_size], [score_size, 0]])

        transform = cv2.getPerspectiveTransform(score_pnts, img_pnts)
        score_t = cv2.warpPerspective(score_mat, transform, (img.shape[1], img.shape[0]))

        res = line_mat * score_t
        return -np.average(res[np.where(res > 255 * 0.1)])
    
    return get_fit_score

def convex_poly_fitted(img, show=True):
    polies = convex_poly(img, False)
    poly = select_corners(img, polies)
    x0 = poly.flatten()
    get_fit_score = get_get_fit_score(img, poly)
    ret = basinhopping(get_fit_score, x0, T=0.1, niter=250, stepsize=3)
    return ret.x.reshape(4, 2), ret.fun

def normalize_corners(v):
    rads = []
    for i in range(4):
        a = v[(i + 1) % 4] - v[i]
        a = a / np.linalg.norm(a)
        cosv = np.dot(a, np.array([1, 0]))
        rads.append(math.acos(cosv))
    left_top = np.argmin(rads)
    return np.roll(v, 4 - left_top, axis=0)
