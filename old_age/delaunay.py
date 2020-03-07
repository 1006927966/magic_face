import cv2
import dlib
import numpy as np
from imutils import face_utils


jaw_point = list(range(0, 17)) + list(range(68,81))
path = '/home/pc/gitcode/magic_face/models/shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)


def get_landmarks(img):
    faces = detector(img, 1)
    shape = predictor(img, faces[0]).parts()
    return np.array([[p.x, p.y] for p in shape])


def draw_convex_hull(img, points, color):
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(img, hull, color=color)


def get_mask(img):
    landmarks = get_landmarks(img)
    mask = np.zeros(img.shape[:2])
    draw_convex_hull(mask, landmarks[jaw_point], color=1)
    mask = np.array([mask] * 3).transpose(1, 2, 0)
    return mask


def triangulate(img):
    faces = detector(img, 1)
    (x, y, w, h) = face_utils.rect_to_bb(faces[0])
    cx = int(.15 * w)
    cy = int(.5 * h)
    subdiv = cv2.Subdiv2D(
        (max(x - cx, 0), max(y - cy, 0), min(w + x + cx, img.shape[1]), min(h + y + cx, img.shape[0])))
    landmarks = get_landmarks(img)
    for x, y in landmarks:
        subdiv.insert((int(x), int(y)))
    return landmarks, subdiv.getTriangleList()


if __name__ == '__main__':
    path = '/home/pc/桌面/1.jpg'
    img = cv2.imread(path)
    a, b = triangulate(img)
    print(a)
    print(b)