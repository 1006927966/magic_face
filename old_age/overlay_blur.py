import cv2


def overlay(img1, img2, mask):
    blur_mask = cv2.blur(mask, (20, 20))
    new = img2.copy()
    for y in range(0, img1.shape[0]):
        for x in range(0, img1.shape[1]):
            w = blur_mask[y][x]/255
            if w > 0.6:
                w = (w-0.6)/0.4
            else:
                w = 0
            new[y][x] = img2[y][x]*w + img1[y][x]*(1-w)
    return new

