import cv2
import numpy as np


def old_pic(img):
    rows, cols, channals = img.shape
    for r in range(rows):
        for c in range(cols):
            B = img.item(r, c, 0)
            G = img.item(r, c, 1)
            R = img.item(r, c, 2)
            img[r, c, 0] = np.uint8(min(max(0.272 * R + 0.534 * G + 0.131 * B, 0), 255))
            img[r, c, 1] = np.uint8(min(max(0.349 * R + 0.686 * G + 0.168 * B, 0), 255))
            img[r, c, 2] = np.uint8(min(max(0.393 * R + 0.769 * G + 0.189 * B, 0), 255))
    return img


def pencil(img):
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=50, sigma_r=0.15, shade_factor=0.04)
    return dst_gray, dst_color


def cartoon(img):
    dst_comic = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    return dst_comic

if __name__ == '__main__':
    path = '/home/pc/face_study/exp/timg.jpg'
    save_path = '/home/pc/桌面/tex/cartoon.jpg'
    img = cv2.imread(path)
    img_color = cartoon(img)
    cv2.imwrite(save_path, img_color)

