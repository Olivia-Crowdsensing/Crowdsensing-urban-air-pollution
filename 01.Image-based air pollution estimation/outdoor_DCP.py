# -*- coding: utf-8 -*-
import cv2
import numpy as np


def DarkChannel(im, size):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    import math

    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)
    indices = np.argsort(darkvec, 0)
    indices = indices[imsz - numpx::]
    b, g, r = cv2.split(im)
    gray_im = r * 0.299 + g * 0.587 + b * 0.114
    gray_im = gray_im.reshape(imsz, 1)
    loc = np.where(gray_im == max(gray_im[indices]))
    x = loc[0][0]
    A = np.array(imvec[x])
    A = A.reshape(1, 3)
    return A


def get_atmosphere(I, darkch):
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()

    searchidx = (-flatdark).argsort()[:10]
    A = np.max(flatI.take(searchidx, axis=0), axis=0)
    return A


def get_dark_channel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), 'edge')

    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_transmission(I, A, darkch, omega):
    w = 7  # window size
    omega = 1
    return 1 - 0.95 * get_dark_channel(I / A, w)  # CVPR09, eq.12


def TransmissionEstimate(im, A, size):
    omega = 1
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]  # individual calculate Te in each channel with normalize
    transmission = 1 - omega * DarkChannel(im3, size)
    return transmission


def calDepthMap(I, r):
    import scipy
    from scipy import ndimage

    hsvI = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    s = hsvI[:, :, 1] / 255.0
    v = hsvI[:, :, 2] / 255.0
    sigma = 0.041337
    sigmaMat = np.random.normal(0, sigma, (I.shape[0], I.shape[1]))
    output = 0.121779 + 0.959710 * v - 0.780245 * s + sigmaMat
    outputPixel = output
    output = scipy.ndimage.filters.minimum_filter(output, (r, r))
    depthmap = output
    return depthmap, outputPixel


def trans_filter(trans):
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            if (trans[i, j] == 0):
                continue
            else:
                trans[i, j] = -1 * np.log((trans[i, j] / 255))
    count_val = 0
    count_num = 0
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            if trans[i, j] == 0 or trans[i, j] == 1:
                continue
            else:
                count_val = count_val + trans[i, j]
                count_num = count_num + 1
    return (count_val / count_num)


# Left things
def extinction(depth, trans):
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            if (trans[i, j] == 0):
                continue
            else:
                trans[i, j] = np.log((trans[i, j] / 255))
    trans_log = trans
    beta = trans

    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            if (depth[i, j] == 0):
                continue
            else:
                beta[i, j] = -1 * (trans_log[i, j] / (depth[i, j] / 255))

    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            if (beta[i, j] < 0):
                beta[i, j] = 0
            else:
                continue
    count_val = 0
    count_num = 0
    for i in range(beta.shape[0]):
        for j in range(beta.shape[1]):
            if (beta[i, j] == 0):
                continue
            else:
                count_val = count_val + beta[i, j]
                count_num = count_num + 1
    bet = count_val / count_num
    return bet
