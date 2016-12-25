import numpy as np
import cv2
import math
import random

def rn():
    return random.random()

brushes = {}

# load brushes from ./brushes directory
def load_brushes():
    brush_dir = './brushes/'
    import os
    for fn in os.listdir(brush_dir):
        if os.path.isfile(brush_dir + fn):
            brush = cv2.imread(brush_dir + fn,0)
            if not brush is None:
                brushes[fn] = brush

load_brushes()

def get_brush(key='random'):
    if key=='random':
        key = random.choice(list(brushes.keys()))
    brush = brushes[key]
    return brush,key

def rotate_brush(brush,rad,srad,angle):
    # brush image should be of grayscale, pointing upwards

    # translate w x h into an area of 2rad x 2rad

    bh,bw = brush.shape[0:2]
    # print(brush.shape)

    osf = 0.1
    # oversizefactor: ratio of dist-to-edge to width,
    # to compensate for the patch smaller than the original ellipse

    rad = int(rad*(1.+osf))
    srad = int(srad*(1.+osf))

    # 1. scale
    orig_points = np.array([[bw/2,0],[0,bh/2],[bw,bh/2]]).astype('float32')
    # x,y of top left right
    translated = np.array([[rad,0],[rad-srad,rad],[rad+srad,rad]]).astype('float32')

    # affine transform matrix
    at = cv2.getAffineTransform(orig_points,translated)

    at = np.vstack([at,[0,0,1.]])

    # 2. rotate
    rm = cv2.getRotationMatrix2D((rad,rad),angle-90,1)
    # per document:
    # angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).

    # stroke image should point eastwards for 0 deg, hence the -90

    rm = np.vstack([rm,[0,0,1.]])

    # 3. combine 2 affine transform
    cb = np.dot(rm,at)
    # print(cb)

    # 4. do the transform
    res = cv2.warpAffine(brush,cb[0:2,:],(rad*2,rad*2))
    return res

def lc(i): #low clip
    return int(max(0,i))

def generate_motion_blur_kernel(dim=3,angle=0.):
    radian = angle/360*math.pi*2 +  math.pi/2
    # perpendicular
    x2,y2 = math.cos(radian),math.sin(radian)
    # the directional vector

    center = (dim+1)/2-1
    kernel = np.zeros((dim,dim)).astype('float32')

    # iterate thru kernel
    ki = np.nditer(kernel,op_flags=['readwrite'],flags=['multi_index'])
    while not ki.finished:
        ya,xa = ki.multi_index
        y1,x1 = -(ya-center),xa-center # flip y since it's image axis

        # now y1, x1 correspond to each kernel pixel's cartesian coord
        # with center being 0,0

        dotp = x1*x2 + y1*y2 # dotprod

        ki[0] = dotp
        ki.iternext()

    kernel = (1-kernel*kernel).clip(min=0)
    kernel /= dim*dim*np.mean(kernel)

    return kernel

# the brush process
def compose(orig,brush,x,y,rad,srad,angle,color,usefloat=False,useoil=False):
    # generate, scale and rotate the brush as needed
    brush_image = rotated = rotate_brush(brush,rad,srad,angle) # as alpha
    brush_image = np.reshape(brush_image,brush_image.shape+(1,)) # cast alpha into (h,w,1)

    # width and height of brush image
    bh = brush_image.shape[0]
    bw = brush_image.shape[1]

    y,x = int(y),int(x)

    # calculate roi params within orig to paint the brush
    ym,yp,xm,xp = y-bh/2,y+bh/2,x-bw/2,x+bw/2

    # w and h of orig
    orig_h,orig_w = orig.shape[0:2]

    #crop the brush if exceed orig or <0
    alpha = brush_image[lc(0-ym):lc(bh-(yp-orig_h)),lc(0-xm):lc(bw-(xp-orig_w))]

    #crop the roi params if < 0
    ym,yp,xm,xp = lc(ym),lc(yp),lc(xm),lc(xp)
    roi = orig[ym:yp,xm:xp]

    # print(alpha.shape,roi.shape)

    if alpha.shape[0]==0 or alpha.shape[1]==0 or roi.shape[0]==0 or roi.shape[1]==0:
        return # dont paint if roi or brush is empty

    # to simulate oil painting mixing:
    # color should blend in some fasion from given color to bg color
    if useoil:
        if usefloat:
            pass
        else:
            roi = roi.astype('float32')/255.
            color = np.array(color).astype('float32')/255.

        alpha = alpha.astype('float32')/255.

        from colormixer import oilpaint_converters
        b2p,p2b = oilpaint_converters()

        # convert into oilpaint space
        roi,color = b2p(roi),b2p(color)

        def getkernelsize(r):
            k = min(55,int(r/4))
            if k%2==0:
                k+=1
            if k<3:
                k+=2
            return k
        sdim = getkernelsize(srad) # determine the blur kernel characteristics
        ldim = getkernelsize(rad)

        #blur brush pattern
        softalpha = cv2.blur(alpha,(sdim,sdim)) # 0-1

        mixing_ratio = np.random.rand(roi.shape[0],roi.shape[1],1)
        # random [0,1] within shape (h,w,1

        # increase mixing_ratio where brush pattern
        # density is lower than 1
        # i.e. edge enhance
        mixing_ratio[:,:,0] += (1-softalpha)*2

        mixing_th = 0.2 # threshold, larger => mix more
        mixing_ratio = mixing_ratio > mixing_th
        # threshold into [0,1]

        # note: mixing_ratio is of dtype bool

        # larger the mixing_ratio, stronger the color
        colormap = roi - roi*mixing_ratio + color*mixing_ratio

        # apply motion blur on the mixed colormap
        kern = generate_motion_blur_kernel(dim=ldim,angle=angle)

        # print(sdim,ldim,kern.shape,colormap.dtype,kern.dtype,mixing_ratio.dtype,roi.dtype,color.dtype)

        colormap = cv2.filter2D(colormap,cv2.CV_32F,kern)

        if usefloat:
            orig[ym:yp,xm:xp] = p2b(roi*(1-alpha)+colormap*(alpha))
        else:
            orig[ym:yp,xm:xp] = p2b(roi*(1-alpha)+colormap*(alpha))*255.
    else:
        # no oil painting
        colormap = np.array(color).astype('float32') # don't blend with bg, just paint fg

        if usefloat:
            # if original image is float
            alpha = alpha.astype('float32')/255.
            orig[ym:yp,xm:xp] = roi*(1-alpha) + colormap*alpha
        else:
            # integer version
            colormap = colormap.astype('uint32')
            roi = roi.astype('uint32')
            # use uint32 to prevent multiplication overflow
            orig[ym:yp,xm:xp] = (roi*(255-alpha) + colormap*alpha)/255

        # painted

def test():
    flower = cv2.imread('flower.jpg')
    fint = flower.copy()
    for i in range(100):
        brush,key = get_brush()
        color = [rn()*255,rn()*255,rn()*255]

        print('integer no oil')
        compose(fint,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
        rad=50,srad=10+20*rn(),angle=rn()*360,color=color,useoil=False)

        print('integer oil')
        compose(fint,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
        rad=50,srad=10+20*rn(),angle=rn()*360,color=color,useoil=True)

        cv2.imshow('integer',fint)
        cv2.waitKey(10)

    floaty = flower.copy().astype('float32')/255.
    for i in range(100):
        brush,key = get_brush()
        color = [rn(),rn(),rn()]

        print('float no oil')
        compose(floaty,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
        rad=50,srad=10+20*rn(),angle=rn()*360,color=color,usefloat=True,useoil=False)

        print('float oil')
        compose(floaty,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
        rad=50,srad=10+20*rn(),angle=rn()*360,color=color,usefloat=True,useoil=True)

        cv2.imshow('float',floaty)
        cv2.waitKey(10)
