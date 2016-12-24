import numpy as np
import cv2
import math
import random
import time

def get_phase(i):
    #grayify
    # igray = np.mean(i,axis=2).astype('uint8')
    igray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).astype('float32')

    #blur
    # igray = cv2.blur(igray,(3,3))

    # xg,yg = cv2.spatialGradient(igray)
    xg = cv2.Sobel(igray,cv2.CV_32F,1,0,ksize=7)
    yg = - cv2.Sobel(igray,cv2.CV_32F,0,1,ksize=7)
    # for some unknown reasons there has to be a minus sign...
    # otherwise the gradient for y is reversed.

    # xg,yg = xg.astype('float32'),yg.astype('float32')

    def c(i):
        return i.clip(min=0)

    def b(i):
        return cv2.blur(i,(5,5))

    # separate
    # xgp = c(xg)
    # xgn = c(-xg)
    #
    # ygp = c(yg)
    # ygn = c(-yg)
    #
    # xgp,xgn,ygp,ygn = b(xgp),b(xgn),b(ygp),b(ygn)
    #
    # xg = xgp-xgn
    # yg = ygp-ygn

    # xg = xg*xg
    # yg = yg*yg
    # xg,yg = b(xg),b(yg)

    phase = cv2.phase(xg,yg)
    # for p in np.nditer(phase,op_flags=['readwrite']):
    #     p[...] = p-math.pi if p>math.pi else p
    #
    # xg = np.sin(phase)
    # yg = np.cos(phase-math.pi/2)
    #
    # xg,yg = b(xg),b(yg)
    #
    # phase = cv2.phase(xg,yg)

    return phase

def test():
    i = cv2.imread('2622s.jpg')

    phase,xg,yg = get_phase(i)

    # color = np.zeros(phase.shape+(3,))
    # color[:,:,1] = xg / np.max(xg)*20
    # color[:,:,2] = yg / np.max(yg)*20

    phase = phase/math.pi/2

    for p in np.nditer(phase,op_flags=['readwrite']):
        p[...] = p-0.5 if p>0.5 else p

    def dist(a1,a2): # both >0 <0.5
        if a2>a1:
            return min(a2-a1,a1+0.5-a2)
        else:
            return min(a1-a2,a2+0.5-a1)

    def surround_dist(y,x,delta=0.0):
        base = phase[y,x]+delta
        if base>0.5:
            base-=0.5

        sd = dist(base,phase[y,x+1])+dist(base,phase[y,x-1])
        +dist(base,phase[y+1,x])+dist(base,phase[y-1,x])
        +dist(base,phase[y+1,x+1])+dist(base,phase[y-1,x+1])
        +dist(base,phase[y+1,x-1])+dist(base,phase[y-1,x-1])
        return sd/8

    def descent():
        for r in range(1,phase.shape[0]-2):
            for c in range(1,phase.shape[1]-2):
                sd = surround_dist(r,c)
                gsd = (surround_dist(r,c,1e-3)-sd)/1e-3
                k = phase[r,c]
                k -= gsd*.1
                if k>0.5:
                    k-=0.5
                if k<0:
                    k+=0.5
                phase[r,c] = k

    descent()

    cv2.imshow('phase',phase)

    cv2.waitKey(100)
