import cv2
import numpy as np
from mask import get_organ_mask , get_landmark, left_eye, right_eye, align


def affine_matrix(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def read_im_and_landmarks(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img.shape[1], img.shape[0]))
    s = get_landmark(img)
    return img, s


def warp_im(img, M, shape):
    output_img = np.zeros(shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (shape[1], shape[0]),
                   dst=output_img,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_img


def correct_colours(im1, im2, landmarks1):
    blur_amount = 0.6 * np.linalg.norm(
                              np.mean(landmarks1[left_eye], axis=0) -
                              np.mean(landmarks1[right_eye], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                                im2_blur.astype(np.float64))



# the source get the target orgin
def swap_orgin(source_path, target_path, tag):
    source, landmark1 = read_im_and_landmarks(source_path)
    target, landmark2 = read_im_and_landmarks(target_path)
    M = affine_matrix(landmark1[align], landmark2[align])
    mask = get_organ_mask(target, tag)
    warp_mask = warp_im(mask, M, source.shape)
    combined_mask = np.max([get_organ_mask(source, tag), warp_mask],
                              axis=0)
    warp_target = warp_im(target, M, source.shape)
    correct_target = correct_colours(source, warp_target, landmark1)
    output_img = source*(1.0-combined_mask) + correct_target*combined_mask
    return output_img


if __name__ == '__main__':
    path1 = '/home/pc/桌面/1.jpg'
    path2 = '/home/pc/face_study/exp/mk.jpg'
    out_img = swap_orgin(path1, path2, 'mouth')
    cv2.imwrite('/home/pc/桌面/tex/mouth.jpg', out_img)
