import cv2
import numpy as np
import dlib

dlib_path = './models/shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_path)

jaw_point = list(range(0, 17)) + list(range(68,81))
left_eye = list(range(42, 48))
right_eye = list(range(36, 42))
left_brow = list(range(22, 27))
right_brow = list(range(17, 22))
mouth = list(range(48, 61))
nose = list(range(27, 35))


align = (left_brow + right_eye + left_eye +
                               right_brow + nose + mouth)


def get_landmark(img):
    faces = detector(img, 1)
    shape = predictor(img, faces[0]).parts()
    return np.matrix([[p.x, p.y] for p in shape])


def draw_convex_hull(img, points, color):
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(img, hull, color=color)


def get_organ_mask(img, tag):
    landmarks = get_landmark(img)
    mask = np.zeros(img.shape[:2])
    if tag == 'eye':
        white = [right_eye, left_eye]
    if tag == 'nose':
        white = [nose]
    if tag == 'mouth':
        white = [mouth]
    if tag == 'eyebrow':
        white = [left_brow, right_brow]
    for group in white:
        points = landmarks[group]
        draw_convex_hull(mask, points, 1)
    mask = np.array([mask]*3).transpose(1, 2, 0)
    mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask


def get_skin_mask(img):
    landmarks = get_landmark(img)
    mask = np.zeros(img.shape[:2])
    draw_convex_hull(mask, landmarks[jaw_point], color=1)
    for index in [mouth, left_eye, right_eye, left_brow, right_brow, nose]:
        draw_convex_hull(mask, landmarks[index], color=0)
    mask = np.array([mask] * 3).transpose(1, 2, 0)
    return mask


if __name__ == '__main__':
    path = '/home/pc/face_study/exp/timg.jpg'
    img = cv2.imread(path)
    mask = get_organ_mask(img, 'eye')
    cv2.imshow('a', mask)
    cv2.waitKey(0)