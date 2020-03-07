import random

import cv2
import numpy as np
from imutils import face_utils
from math import e, sqrt, pi
from delaunay import path
import dlib


def lip_makeup(subject, warped_target):

    gray_sub = cv2.cvtColor(subject, cv2.COLOR_BGR2GRAY)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)

    # detect faces in the grayscale image
    rects = detector(gray_sub, 1)

    upperlip_ind = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60]
    lowerlip_ind = [48, 60, 67, 66, 65, 64, 53, 55, 56, 57, 58, 59]
    lip_pts = []


    lip_map = np.zeros(subject.shape, dtype=subject.dtype)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray_sub, rect)
        shape = face_utils.shape_to_np(shape)
        for x in range (48, 62):
            lip_pts.append(shape[x])
        C2 = cv2.convexHull(np.array(lip_pts))
        cv2.drawContours(lip_map, [C2], -1, (255, 255, 255), -1)

        lip_pts = []
        for x in range (60, 67):
            lip_pts.append(shape[x])
        C2 = cv2.convexHull(np.array(lip_pts))
        cv2.drawContours(lip_map, [C2], -1, (0, 0, 0), -1)

    #cv2.imshow('s', subject)
    #cv2.imshow('t', warped_target)
    #cv2.imshow('lip map', lip_map)
    #cv2.imwrite('add', np.where(not lip_map[:] == [0, 0, 0], lip_map, subject))
    overlay = subject.copy()
    overlay = np.where(lip_map != [0, 0, 0], lip_map, overlay)
    #cv2.imshow('overlay', overlay)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



    l_E , _, _ = cv2.split(cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB))
    l_I , _, _ = cv2.split(cv2.cvtColor(subject, cv2.COLOR_BGR2LAB))

    print('Histogram remapping for reference image \'E\' ...')

    l_E_sum = 0
    l_E_sumsq = 0
    l_I_sum = 0
    l_I_sumsq = 0
    lip_pts = []
    for y in range(0, lip_map.shape[0]):
        for x in range(0, lip_map.shape[1]):
            # print(lip_map[y][x])
            if (lip_map[y][x][2] != 0):
                l_E_sum += l_E[y, x]    #calculating mean for only lip area
                l_E_sumsq += l_E[y, x]**2    #calculating var for only lip area
                l_I_sum += l_I[y, x]    #calculating mean for only lip area
                l_I_sumsq += l_I[y, x]**2    #calculating var for only lip area
                lip_pts.append([y, x])

    print(len(lip_pts))

    l_E_mean = l_E_sum / len(lip_pts)
    l_I_mean = l_I_sum / len(lip_pts)

    l_E_std = sqrt((l_E_sumsq / len(lip_pts)) - l_E_mean**2)
    l_I_std = sqrt((l_I_sumsq / len(lip_pts)) - l_I_mean**2)

    l_E = (l_I_std / l_E_std * (l_E - l_E_mean)) + l_I_mean  # fit the hostogram of source to match target(imgI) Luminance remapping



    def Gauss(x):
        return e ** (-0.5 * float(x))

    M = cv2.cvtColor(subject, cv2.COLOR_BGR2LAB)
    warped_target_LAB = cv2.cvtColor(warped_target, cv2.COLOR_BGR2LAB)
    counter = 0

    sample = lip_pts.copy()
    random.shuffle(sample)
    avg_maxval = 0
    for p in lip_pts:
        q_tilda = 0
        maxval = -1
        counter += 1
        print(counter / len(lip_pts) * 100, " %")
        for i in range(0, 500):
            q = sample[i]
            curr = (Gauss(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / 5) * Gauss(((float(l_E[q[0]][q[1]]) - float(l_I[p[0]][p[1]])) / 255) ** 2))
            if maxval < curr:
                maxval = curr
                q_tilda = q
                if maxval >= 0.9:
                    break

        avg_maxval += maxval
        print("max = ", maxval)
        M[p[0]][p[1]] = warped_target_LAB[q_tilda[0]][q_tilda[1]]
    #cv2.imshow('M', cv2.cvtColor(M, cv2.COLOR_LAB2BGR))
    print("avgmax = ", avg_maxval/len(lip_pts))

    output = cv2.cvtColor(subject.copy(), cv2.COLOR_BGR2LAB)
    for p in lip_pts:
        output[p[0]][p[1]][1] = M[p[0]][p[1]][1]
        output[p[0]][p[1]][2] = M[p[0]][p[1]][2]

    output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
    #cv2.imshow('out', output)
    #cv2.waitKey(0)
    #cv2.imwrite('LipOut.jpg', output)
    return output, lip_map
