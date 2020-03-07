import cv2
import numpy as np
from mask import get_skin_mask
from mask import get_organ_mask


def buff(img, img_skin, value1, value2):
    dx = value1*5
    fc = value1*12.5
    p = 80
    temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    temp2 = (temp1-img+128)
    temp3 = cv2.GaussianBlur(temp2, (2*value2-1, 2*value2-1), 0, 0)
    temp4 = img+2*temp3-255
    dst = np.uint8(img*((100-p)/100) + temp4*(p/100))
    img_skin_c = np.uint8(-(img_skin - 1))
    dst = np.uint8(dst*img_skin+img*img_skin_c)
    return dst


def whitening(img, img_skin, value):
    midtones_add = np.zeros(256)
    for i in range(256):
        midtones_add[i] = 0.667*(1-((i-127)/127)*((i-127)/127))
    lookup = np.zeros(256, dtype='uint8')
    for i in range(256):
        red = i
        red += np.uint8(value*midtones_add[red])
        red = max(0, red)
        lookup[i] = np.uint(red)

    w, h, c = img.shape
    for i in range(w):
        for j in range(h):
            if img_skin[i, j, 0] == 1:
                img[i, j, 0] = lookup[img[i, j, 0]]
                img[i, j, 1] = lookup[img[i, j, 1]]
                img[i, j, 2] = lookup[img[i, j, 2]]
    return img


def light(img, value, eye_mask):
    w, h = eye_mask.shape[:2]
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    for i in  range(w):
        for j in range(h):
            if eye_mask[i, j, 0] == 1:
                if value > 3:
                    img_hls[i, j, 1] = np.log(img_hls[i, j, 1] / 255 * (value - 1) + 1) / np.log(value + 1) * 255
                elif value < 0:
                    img_hls[i, j, 1] = np.uint8(img_hls[i, j, 1] / np.log(-value + np.e))

    light_img = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    return light_img


if __name__ == '__main__':
     path = '/home/pc/桌面/1.jpg'
     img = cv2.imread(path)
     skin_mask = get_skin_mask(img)
     eye_mask = get_organ_mask(img, 'eye')
     dst = light(img, 5, eye_mask)
     cv2.imwrite('/home/pc/桌面/tex/light.jpg', dst)
