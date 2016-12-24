import numpy as np
import cv2
import math
import random
import time

# calculate phase angle from image gradient
# in float32 radian
def get_phase(i):
    # grayify
    igray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY).astype('float32')

    # gradient
    xg = cv2.Sobel(igray,cv2.CV_32F,1,0,ksize=7)
    yg = - cv2.Sobel(igray,cv2.CV_32F,0,1,ksize=7)
    # in image axis y points downwards, hence the minus sign

    def c(i):
        return i.clip(min=0)

    def b(i):
        return cv2.blur(i,(5,5))

    phase = cv2.phase(xg,yg)

    return phase # radian

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
