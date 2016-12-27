print('importing packages...')
import numpy as np
import cv2
import math
import random
import time
import rotate_brush as rb
import gradient
from thready import amap
import os
import threading

canvaslock = threading.Lock()
canvaslock.acquire()
canvaslock.release()

def lockgen(canvas,ym,yp,xm,xp):
    # given roi, know which lock.
    #
    # if left:
    #     return leftcanvaslock:
    # if right:
    #     return rightcanvaslock:
    # if riding:
    #     reutrn canvaslock:
    pass
def load(filename='flower.jpg'):
    print('loading',filename,'...')
    global imname,flower,canvas,hist
    global rescale,xs_small,ys_small,smallerflower

    imname = filename.split('.')[0]

    # original image
    flower = cv2.imread(filename)

    xshape = flower.shape[1]
    yshape = flower.shape[0]

    rescale = xshape/640
    # display rescaling: you'll know when it's larger than your screen
    if rescale<1:
        rescale=1

    xs_small = int(xshape/rescale)
    ys_small = int(yshape/rescale)

    smallerflower = cv2.resize(flower,dsize=(xs_small,ys_small)).astype('float32')/255
    # for preview purpose,
    # if image too large

    # convert to float32
    flower = flower.astype('float32')/255

    # canvas initialized
    canvas = flower.copy()
    canvas[:,:] = 0.8

    #clear hist
    hist=[]
    print(filename,'loaded.')

load()

def rn():
    return random.random()

def showimg():
    if rescale==1:
        smallercanvas = canvas
    else:
        smallercanvas = cv2.resize(canvas,dsize=(xs_small,ys_small),interpolation=cv2.INTER_NEAREST)

    i,j,d = wherediff(smallercanvas,smallerflower)
    sd = np.mean(d)
    print('mean diff:',sd)

    d[i,:]=1.0
    d[:,j]=1.0

    cv2.imshow('canvas',smallercanvas)
    cv2.imshow('flower',smallerflower)
    cv2.imshow('diff',d)

    cv2.waitKey(1)
    cv2.waitKey(1)

def destroy():
    cv2.destroyAllWindows()

def positive_sharpen(i,overblur=False,coeff=8.): #no darken to original image
    # emphasize the edges
    blurred = cv2.blur(i,(5,5))
    sharpened = i + (i - blurred) * coeff
    if overblur:
        return cv2.blur(np.maximum(sharpened,i),(11,11))
    return cv2.blur(np.maximum(sharpened,i),(3,3))

def diff(i1,i2,overblur=False):
    #calculate the difference of 2 float32 BGR images.

    # # use lab
    # i1=i1.astype(np.float32)
    # i2=i2.astype(np.float32)
    # lab1 = cv2.cvtColor(i1,cv2.COLOR_BGR2LAB)
    # lab2 = cv2.cvtColor(i2,cv2.COLOR_BGR2LAB)
    # d = lab1-lab2
    # d = d*d / 10000

    # # use rgb
    d = (i1-i2)# * [0.2,1.5,1.3]
    d = d*d

    d = positive_sharpen(np.sum(d,-1),overblur=overblur)
    return d
    # grayscalize

def wherediff(i1=None,i2=None):
    global canvas,flower
    if i1 is None:
        i1 = canvas
    if i2 is None:
        i2 = flower

    # find out where max difference point is.
    d = diff(i1,i2,overblur=True)

    i,j = np.unravel_index(d.argmax(),d.shape)
    return i,j,d

def get_random_color():
    return np.array([rn(),rn(),rn()]).astype('float32')
    #danger: default to float64

def limit(x,minimum,maximum):
    return min(max(x,minimum),maximum)

# history and replay section

# global history.
hist = []
def record(sth):
    hist.append(sth)

# repaint the image from history
def repaint(constraint_angle=False,upscale=1.,batchsize=16):
    starttime = time.time()

    newcanvas = np.array(canvas).astype('uint8')
    # newcanvas = cv2.cvtColor(newcanvas,cv2.COLOR_BGR2BGRA) # fastest format

    if upscale!=1.:
        newcanvas = cv2.resize(newcanvas,dsize=(int(newcanvas.shape[1]*upscale),int(newcanvas.shape[0]*upscale)))

    newcanvas[:,:,:] = int(0.8*255)

    def showthis():
        showsize = 640
        resize_scale = min(showsize/newcanvas.shape[1],1.)
        resizedx,resizedy = int(newcanvas.shape[1]*resize_scale),int(newcanvas.shape[0]*resize_scale)

        smallercanvas = cv2.resize(newcanvas,dsize=(resizedx,resizedy),interpolation=cv2.INTER_NEAREST)
        cv2.imshow('repaint',smallercanvas)
        cv2.waitKey(1)

    def paintone(histitem):
        x,y,radius,srad,angle,cb,cg,cr,brushname = histitem

        cb,cg,cr = int(cb*255),int(cg*255),int(cr*255)
        # cv2.ellipse(newcanvas,(int(x),int(y)),(radius,srad),angle,0,360,color=(cb,cg,cr),thickness=-1)

        b,key = rb.get_brush(brushname)

        if constraint_angle:
            angle = constraint_angle+rn()*20-10

        if upscale!=1:
            x,y,radius,srad = x*upscale,y*upscale,radius*upscale,srad*upscale

        rb.compose(newcanvas,b,x=x,y=y,rad=radius,srad=srad,angle=angle,color=[cb,cg,cr],useoil=True,lock=canvaslock)

    k = 0
    batch = []

    def runbatch(batch):
        from thready import amap # multithreading
        return amap(paintone,batch)

    lastep = 0

    while k<len(hist):
        while len(batch)<batchsize and k<len(hist):
            batch.append(hist[k])
            k+=1
        runbatch(batch)
        print(k,'painted. one of them:',batch[0])

        # show progress:
        ep = int(k/(newcanvas.shape[1]*upscale)) # larger image => longer wait per show
        if ep >lastep:
            showthis()
            lastep = ep # show every 32p

        batch=[]

    print(time.time()-starttime,'s elapsed')
    showthis()
    return newcanvas

import json
def savehist(filename='hist.json'):
    f = open(filename,'w')
    json.dump(hist,f)
    f.close()

def loadhist(filename='hist.json'):
    f = open(filename,'r')
    global hist
    hist = json.load(f)

# end hist section

def paint_one(x,y,brushname='random',angle=-1.,minrad=10,maxrad=60):
    oradius = rn()*rn()*maxrad+minrad
    fatness = 1/(1+rn()*rn()*6)

    brush,key = rb.get_brush(brushname)

    def intrad(orad):
        #obtain integer radius and shorter-radius
        radius = int(orad)
        srad = int(orad*fatness+1)
        return radius,srad

    radius,srad = intrad(oradius)

    #set initial angle
    if angle == -1.:
        angle = rn()*360

    # set initial color
    # c = get_random_color()
    # sample color from image => converges faster.
    c = flower[int(y),int(x),:]

    delta = 1e-4

    # get copy of square ROI area, to do drawing and calculate error.
    def get_roi(newx,newy,newrad):
        radius,srad = intrad(newrad)

        xshape = flower.shape[1]
        yshape = flower.shape[0]


        yp = int(min(newy+radius,yshape-1))
        ym = int(max(0,newy-radius))
        xp = int(min(newx+radius,xshape-1))
        xm = int(max(0,newx-radius))

        if yp<=ym or xp<=xm:
            # if zero w or h
            raise NameError('zero roi')

        ref = flower[ym:yp,xm:xp]
        bef = canvas[ym:yp,xm:xp]
        aftr = np.array(bef)

        # print(flower.dtype,canvas.dtype,ref.dtype)
        return ref,bef,aftr

    # paint one stroke with given config and return the error.
    def paint_aftr_w(color,angle,nx,ny,nr):
        ref,bef,aftr = get_roi(nx,ny,nr)
        radius,srad = intrad(nr)

        # cv2.circle(aftr,(radius,radius),radius,color=color,thickness=-1)
        # cv2.ellipse(aftr,(radius,radius),(radius,srad),angle,0,360,color=color,thickness=-1)

        rb.compose(aftr,brush,x=radius,y=radius,rad=radius,srad=srad,angle=angle,color=color,usefloat=True,useoil=False)
        # if useoil here set to true: 2x slow down + instability

        err_aftr = np.mean(diff(aftr,ref))
        return err_aftr

    # finally paint the same stroke onto the canvas.
    def paint_final_w(color,angle,nr):
        radius,srad = intrad(nr)

        # cv2.circle(canvas,(x,y), radius, color=color,thickness=-1)
        # cv2.ellipse(canvas,(int(x),int(y)),(radius,srad),angle,0,360,color=color,thickness=-1)

        rb.compose(canvas,brush,x=x,y=y,rad=radius,srad=srad,angle=angle,color=color,usefloat=True,useoil=True,lock=canvaslock)
        # enable oil effects on final paint.

        # np.float64 will cause problems
        rec = [x,y,radius,srad,angle,color[0],color[1],color[2],brushname]
        rec = [float(r) if type(r)==np.float64 or type(r)==np.float32 else r for r in rec]
        record(rec)
        # log it!

    # given err, calculate gradient of parameters wrt to it
    def calc_gradient(err):
        b,g,r = c[0],c[1],c[2]
        cc = b,g,r

        err_aftr = paint_aftr_w((b+delta,g,r),angle,x,y,oradius)
        gb = err_aftr - err

        err_aftr = paint_aftr_w((b,g+delta,r),angle,x,y,oradius)
        gg = err_aftr - err

        err_aftr = paint_aftr_w((b,g,r+delta),angle,x,y,oradius)
        gr = err_aftr - err

        err_aftr = paint_aftr_w(cc,(angle+5.)%360,x,y,oradius)
        ga = err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x+2,y,oradius)
        gx =  err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x,y+2,oradius)
        gy =  err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x,y,oradius+3)
        gradius = err_aftr - err

        return np.array([gb,gg,gr])/delta,ga/5,gx/2,gy/2,gradius/3,err

    # max and min steps for gradient descent
    tryfor = 12
    mintry = 3

    for i in range(tryfor):
        try: # might have error
            # what is the error at ROI?
            ref,bef,aftr = get_roi(x,y,oradius)
            orig_err = np.mean(diff(bef,ref))

            # do the painting
            err = paint_aftr_w(c,angle,x,y,oradius)

            # if error decreased:
            if err<orig_err and i>=mintry :
                paint_final_w(c,angle,oradius)
                return True,i

            # if not satisfactory
            # calculate gradient
            grad,anglegrad,gx,gy,gradius,err = calc_gradient(err)

        except NameError as e:
            print(e)
            print('error within calc_gradient')
            return False,i

        if printgrad: #debug purpose.
            if i==0:
                print('----------')
                print('orig_err',orig_err)
            print('ep:{}, err:{:3f}, color:{}, angle:{:2f}, xy:{:2f},{:2f}, radius:{:2f}'.format(i,err,c,angle,x,y,oradius))

        # do descend
        if i<tryfor-1:
            c = c - (grad*.3).clip(max=0.3,min=-0.3)
            c = c.clip(max=1.,min=0.)
            angle = (angle - limit(anglegrad*100000,-5,5))%360
            x = x - limit(gx*1000*radius,-3,3)
            y = y - limit(gy*1000*radius,-3,3)
            oradius = oradius* (1-limit(gradius*20000,-0.2,.2))
            oradius = limit(oradius,7,100)

            # print('after desc:x:{:2f},y:{:2f},angle:{:2f},oradius:{:5f}'
            # .format(x,y,angle,oradius))

    return False,tryfor

def putstrokes(howmany):

    def samplepoints():
        # sample a lot of points from one error image - save computation cost

        point_list = []
        y,x,d = wherediff()
        phasemap = gradient.get_phase(flower)

        # while not enough points:
        while len(point_list)<howmany:
            # randomly pick one point
            yshape,xshape = flower.shape[0:2]
            ry,rx = int(rn()*yshape),int(rn()*xshape)

            # accept with high probability if error is large
            # and vice versa
            if d[ry,rx]>0.5*rn():
                # get gradient orientation info from phase map
                phase = phasemap[ry,rx] # phase should be between [0,2pi)

                # choose direction perpendicular to gradient
                angle = (phase/math.pi*180+90)%360
                # angle = 22.5

                point_list.append((ry,rx,angle))
        return point_list

    def pcasync(tup):
        y,x,angle = tup

        b,key = rb.get_brush(key='random') # get a random brush
        return paint_one(x,y,brushname=key,minrad=10,maxrad=50,angle=angle) #num of epoch

    if True:
        from thready import amap # multithreading
        point_list = samplepoints()
        return amap(pcasync,point_list)

    else: # single threading test
        point_list = samplepoints()
        res={}
        for idx,item in enumerate(point_list):
            print('single threaded mode.',idx)
            res[idx] = pcasync(item)
        return res

# autosave during canvas painting
dosaveimage = True
# dosaveimage = False

# gradient debug info print
printgrad = False
# printgrad = True

# run the whole thing
def r(epoch=1):
    # filename prefix for each run
    seed = int(rn()*1000)

    print('running...')
    st = time.time()

    # timing counter for autosave and showimg()
    timecounter = 0
    showcounter = 0

    for i in range(epoch):
        loopfor = 1
        paranum = 256
        # number of stroke tries per batch, sent to thread pool
        # smaller number decreases efficiency

        succeeded = 0 # how many strokes being placed
        ti = time.time()

        # average step of gradient descent performed
        avgstep=0.
        for k in range(loopfor):
            res = putstrokes(paranum) # res is a map of results

            for r in res:
                status,step = res[r]
                avgstep += step
                succeeded += 1 if status else 0

        avgstep/=loopfor*paranum

        steptime = time.time()-ti
        tottime = time.time()-st

        #info out
        print('epoch',i,'/',epoch ,'succeeded:',succeeded,'/',loopfor*paranum,'avg step:' ,avgstep,'time:{:.1f}s, total:{:.1f}s'.format(steptime,tottime))

        #autosave
        timecounter+=steptime
        if(timecounter>20):
            timecounter=0
            if dosaveimage:
                print('saving to disk...')

                if not os.path.exists('./'+imname):
                    os.mkdir('./'+imname)

                cv2.imwrite(imname+'/{}_{:04d}.png'.format(seed,i),canvas*255)
                print('saved.')

        # refresh view
        showcounter+=steptime
        if(showcounter>3):
            showcounter=0
            showimg()
    showimg()
