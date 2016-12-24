# some cv2 shit.

import cv2
import numpy as np
import tensorflow as tf
import time
import keras
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

print('import complete. loading model...')
trained = load_model('evenbetter.h5')
trained.summary()

def rebuild():
    model = Sequential()

    model.add(Convolution2D(32, 7, 7, #border_mode='same',
                            input_shape=(240,320,1)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 7, 7))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))

    # this should cover 32x32 area

    model.add(Convolution2D(16, 1, 1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(1, 1, 1))
    model.add(Activation('relu'))

    model.summary()

    model.set_weights(trained.get_weights())
    print('weight sucessfully set.')
    return model

model = rebuild()

def predict(xi):
    i = np.reshape(xi,(1,240,320,1))
    i = i.astype('float32')
    i/=255
    i-=0.5
    return model.predict(i)

cap = cv2.VideoCapture(0)

sess = tf.Session()
# tf section
tfi = tf.placeholder(tf.float32,(240,320,3))
tfo = tfi/255
tfo = tf.nn.avg_pool([tfo],ksize=[1,32,32,1],strides=[1,16,16,1],padding='SAME')

def mainloop():
    lastelapsed = .1
    while(True):


        ret,frame = cap.read()
        if ret != True:
            print(ret)
            break

        starttime = time.time()

        frame = cv2.resize(frame,dsize=(320,240))

        eff = frame

        eff = sess.run(tfo,feed_dict={tfi:eff})[0]

        eff = cv2.resize(eff,dsize=(320,240),interpolation=cv2.INTER_NEAREST)

        eff = cv2.cvtColor(eff,cv2.COLOR_BGR2GRAY)

        # prediction = predict(eff)[0] / 100.0
        #
        # prediction = cv2.resize(prediction,dsize=(320-28,240-28),interpolation=cv2.INTER_NEAREST)
        # heat = cv2.applyColorMap(prediction,cv2.COLORMAP_HOT)

        elapsed = time.time()-starttime
        elapsed = elapsed*.1+lastelapsed*.9
        cv2.putText(frame, "{} ms".format(int(elapsed*1000)),
    		(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        lastelapsed = elapsed


        cv2.imshow('capture',frame)
        cv2.imshow('effect',eff)
        # cv2.imshow('prediction',heat)

        if cv2.waitKey(1)& 0xff == ord('q'):
            break # if q is the pressed key, then exit loop

    cap.release()

    cv2.destroyAllWindows()

mainloop()
