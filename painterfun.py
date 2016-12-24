import numpy as np
import cv2
import math
import random
import time

imname = 'flower'
# flower = cv2.imread('2622.jpg')
flower = cv2.imread(imname+'.jpg')
xshape = flower.shape[1]
yshape = flower.shape[0]

rescale = xshape/768
if rescale<1:
    rescale=1
xs_small = int(xshape/rescale)
ys_small = int(yshape/rescale)

smallerflower = cv2.resize(flower,dsize=(xs_small,ys_small)).astype('float32')/255
# for preview purpose if image too large

flower = np.array(flower).astype('float32')/255

canvas = np.zeros((yshape,xshape,3))+0.8

# mask = np.zeros((yshape,xshape))
# brush = cv2.imread('brush.png',0)
# brush2 = cv2.imread('brush2.png',0)
# brush_leaky = cv2.imread('brush_leaky.png',0)
# brush_s = cv2.imread('brush_s.png',0)
# brush_curvy = cv2.imread('brush_curvy.png',0)
import rotate_brush as rb

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
    # cv2.imshow('mask',mask)
    cv2.waitKey(1)
    cv2.waitKey(1)

def destroy():
    cv2.destroyAllWindows()

def positive_sharpen(i,overblur=False,coeff=8.): #no darken to original image
    # sharpen version of i
    blurred = cv2.blur(i,(5,5))
    sharpened = i*(1+coeff) - blurred * coeff
    if overblur:
        return cv2.blur(np.maximum(sharpened,i),(11,11))
    return cv2.blur(np.maximum(sharpened,i),(3,3))

def diff(i1,i2,overblur=False):
    #calculate lab difference

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

def wherediff(i1=canvas,i2=flower):
    # mask out hot area
    d = diff(i1,i2,overblur=True)

    i,j = np.unravel_index(d.argmax(),d.shape)
    # print('wherediff:',i,j)
    return i,j,d

def getcolor(i,j):
    return flower[i,j,:]

def get_random_color():
    return np.array([rn(),rn(),rn()])

def limit(x,minimum,maximum):
    return min(max(x,minimum),maximum)

# hist replay section

hist = []
def record(sth):
    hist.append(sth)

def repaint(constraint_angle=False,upscale=1.,batchsize=16):
    starttime = time.time()

    newcanvas = np.array(canvas).astype('uint8')
    # newcanvas = cv2.cvtColor(newcanvas,cv2.COLOR_BGR2BGRA) # fastest format

    if upscale!=1.:
        newcanvas = cv2.resize(newcanvas,dsize=(int(newcanvas.shape[1]*upscale),int(newcanvas.shape[0]*upscale)))

    newcanvas[:,:,:] = int(0.8*255)
    # newcanvas[:,:,3] = 255

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
        # b = brush if rn()>0.5 else brush2

        if constraint_angle:
            angle = constraint_angle+rn()*20-10

        if upscale!=1:
            x,y,radius,srad = x*upscale,y*upscale,radius*upscale,srad*upscale

        rb.compose(newcanvas,b,x=x,y=y,rad=radius,srad=srad,angle=angle,color=[cb,cg,cr],useoil=True)

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

        ep = int(k/(newcanvas.shape[1]*upscale)) # larger image => longer wait per show
        if ep >lastep:
            showthis()
            lastep = ep # show every 32p

        batch=[]

    print(time.time()-starttime,'s elapsed')
    showthis()
    return newcanvas

import json
def savehist():
    f = open('hist.json','w')
    json.dump(hist,f)
    f.close()

def loadhist():
    f = open('hist.json','r')
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

    #set initial color
    c = get_random_color()
    c[:] = flower[int(y),int(x),:]# + c[:]*.15

    delta = 1e-4

    # set initial roi
    # yp = int(min(y+radius,yshape-1))
    # ym = int(max(0,y-radius))
    # xp = int(min(x+radius,xshape-1))
    # xm = int(max(0,x-radius))
    # ref = flower[ym:yp,xm:xp]
    # bef = canvas[ym:yp,xm:xp]
    # aftr = np.array(bef)
    # orig_err = np.mean(diff(bef,ref))

    def get_roi(newx,newy,newrad):

        # nonlocal yp,ym,xp,xm
        # if y!=newy or x!=newx or newrad!=oradius:
        # if changes happened to roi

        radius,srad = intrad(newrad)

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

            # orig_err = np.mean(diff(bef,ref))

        # else:
        #     nonlocal aftr,bef
        #     aftr[:] = bef[:]
        return ref,bef,aftr

    def paint_aftr_w(color,angle,nx,ny,nr):
        ref,bef,aftr = get_roi(nx,ny,nr)
        radius,srad = intrad(nr)
        # cv2.circle(aftr,(radius,radius),radius,color=color,thickness=-1)

        # cv2.ellipse(aftr,(radius,radius),(radius,srad),angle,0,360,color=color,thickness=-1)

        rb.compose(aftr,brush,x=radius,y=radius,rad=radius,srad=srad,angle=angle,color=color,usefloat=True,useoil=False)

        # if useoil here set to true: 2x slow down + instability

        err_aftr = np.mean(diff(aftr,ref))
        return err_aftr

    def paint_final_w(color,angle,nr):
        radius,srad = intrad(nr)
        # cv2.circle(canvas,(x,y), radius, color=color,thickness=-1)
        # cv2.ellipse(canvas,(int(x),int(y)),(radius,srad),angle,0,360,color=color,thickness=-1)

        rb.compose(canvas,brush,x=x,y=y,rad=radius,srad=srad,angle=angle,color=color,usefloat=True,useoil=True)
        # enable oil effects on final paint.

        record([x,y,radius,srad,angle,color[0],color[1],color[2],brushname])
        # log it!

    def calc_gradient(err):
        b,g,r = c[0],c[1],c[2]
        cc = b,g,r

        # paint_aftr_w(cc,angle,x,y,oradius)
        # err = err_aftr()
        # print('err',err)
        err_aftr = paint_aftr_w((b+delta,g,r),angle,x,y,oradius)
        gb = err_aftr - err

        err_aftr = paint_aftr_w((b,g+delta,r),angle,x,y,oradius)
        gg = err_aftr - err

        err_aftr = paint_aftr_w((b,g,r+delta),angle,x,y,oradius)
        gr = err_aftr - err

        err_aftr = paint_aftr_w(cc,(angle+5)%360,x,y,oradius)
        ga = err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x+2,y,oradius)
        gx =  err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x,y+2,oradius)
        gy =  err_aftr - err

        err_aftr = paint_aftr_w(cc,angle,x,y,oradius+3)
        gradius = err_aftr - err

        return np.array([gb,gg,gr])/delta,ga/5,gx/2,gy/2,gradius/3,err

    tryfor = 12
    mintry = 1
    for i in range(tryfor):
        try:
            ref,bef,aftr = get_roi(x,y,oradius)

            orig_err = np.mean(diff(bef,ref))

            # paint
            err = paint_aftr_w(c,angle,x,y,oradius)

            # if error is satisfactory
            if err<orig_err and i>=mintry :
                paint_final_w(c,angle,oradius)
                return True,i

        # if not satisfactory

            grad,anglegrad,gx,gy,gradius,err = calc_gradient(err)
        except NameError as e:
            print(e)
            print('error within calc_gradient')
            return False,i

        if printgrad:
            if i==0:
                print('----------')
                print('orig_err',orig_err)
            print('ep:{}, err:{:3f}, color:{}, angle:{:2f}, xy:{:2f},{:2f}, radius:{:2f}'.format(i,err,c,angle,x,y,oradius))

        #descend
        if i<tryfor-1:
            c = c - (grad*.3).clip(max=0.3,min=-0.3)
            c = c.clip(max=1.,min=0.)
            angle = (angle - limit(anglegrad*100000,-5,5))%360
            x = x - limit(gx*1000*radius,-3,3)
            y = y - limit(gy*1000*radius,-3,3)
            oradius = oradius* (1-limit(gradius*20000,-0.2,.2))
            oradius = limit(oradius,7,100)

    return False,tryfor
    # cv2.circle(aftr,(radius,radius),radius,color=c,thickness=-1)
    #
    # if diff(aftr,ref).sum()<orig_err:
        #if after is better
        # cv2.circle(canvas,(x,y), radius, color=c,thickness=-1)

    # global mask
    # paint white on mask
    # cv2.circle(mask,(x,y), int(radius), color=1,thickness=-1)
    # decay the mask
    # mask = mask*0.999

import gradient

def putstrokes(howmany):
    # k is epoch

    # if k<20:
    #     return paintcircle(x,y,minrad=50,maxrad=200) #num of epoch
    # if k<60:
    #     return paintcircle(x,y,minrad=30,maxrad=100) #num of epoch
    # else:
    #     return paintcircle(x,y,minrad=10,maxrad=50) #num of epoch

    def samplepoints():
        # sample a lot of points from one error image - save computation cost

        point_list = []
        y,x,d = wherediff()
        phasemap = gradient.get_phase(flower*255)

        # while not enough points:
        while len(point_list)<howmany:
            # randomly pick one point
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

        b,key = rb.get_brush(key='random') # get random brush
        return paint_one(x,y,brushname=key,minrad=10,maxrad=50,angle=angle) #num of epoch

    from thready import amap # multithreading
    point_list = samplepoints()
    return amap(pcasync,point_list)

dosaveimage = True
# dosaveimage = False

printgrad = False
# printgrad = True

def r(epoch=1):
    seed = int(rn()*1000)
    print('running...')
    st = time.time()
    timecounter = 0
    showcounter = 0
    for i in range(epoch):
        loopfor = 1
        paranum = 512

        succeeded=0
        ti = time.time()

        avgstep=0.
        for k in range(loopfor):
            res = putstrokes(paranum) # res is a map of results
            # print(res)
            for r in res:
                status,step = res[r]
                avgstep += step
                succeeded+= 1 if status else 0

        avgstep/=loopfor*paranum

        steptime = time.time()-ti
        tottime = time.time()-st
        print('epoch',i,'/',epoch ,'succeeded:',succeeded,'/',loopfor*paranum,'avg step:' ,avgstep,'time:{:.1f}s, total:{:.1f}s'.format(steptime,tottime))

        timecounter+=steptime
        if(timecounter>20):
            timecounter=0
            if dosaveimage:
                print('saving to disk...')
                cv2.imwrite(imname+'/{}_{:04d}.png'.format(seed,i),canvas*255)
                print('saved.')

        showcounter+=steptime
        if(showcounter>3):
            showcounter=0
            showimg()
    showimg()
