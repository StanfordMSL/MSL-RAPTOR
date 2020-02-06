#!/usr/bin/env python3

from image_segmentor import ImageSegmentor
import cv2
import numpy as np
import time
import pdb

print("Hi")
im = cv2.imread('SiamMask/data/tennis/00000.jpg')

i_s = ImageSegmentor(im, detect_class_ids=[0, 80])

N = 100

d_tm = 0
t_tm = 0
for i in range(N):
    t0 = time.time()
    det = i_s.detect(im)
    if det is False:
        raise RuntimeError("DETECT FAILED")

    i_s.reinit_tracker(det,im)
    d_tm += time.time() - t0

    t0 = time.time()
    trck = i_s.track(im)
    t_tm += time.time() - t0

print("ave detect time = {:.4f}".format(d_tm / N))
print("ave track time = {:.4f}".format(t_tm / N))

print(det)
print(trck)
