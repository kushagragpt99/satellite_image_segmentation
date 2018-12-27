from __future__ import print_function
from six.moves import range     
from matplotlib import pyplot as plt
import os
from libtiff import TIFF as t
import glob
from keras.models import Sequential
import numpy as np
import sys
import numpy as np
import pandas as pd
import random
import cv2
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Cropping2D, concatenate, ZeroPadding2D, Dense
from keras.optimizers import Adam, Nadam
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
#from keras_metrics import precision, recall
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
import collections

path = os.getcwd()

#gnd_path = path + '/gt/'
#img_path = path + '/sat/'
path = os.getcwd()
gnd_path = path + '/gt/'
img_path = path + '/sat/'
gnd_save_path = gnd_path + 'rotated/'
img_save_path = img_path + 'rotated/'

def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



Roads = [0,0,0]
Water = [0, 0, 150]
Trees = [0, 125, 0]
Grass = [0, 255, 0]
Buildings = [100, 100, 100]
Bare_soil = [150, 80, 0]
Rails = [255, 255, 0]
Pools = [150, 150, 255]
Unlabelled = [255, 255, 255]

color_dict=np.array([Roads,
          Water,
          Trees,
          Grass,
          Buildings,
          Bare_soil,
          Rails,
          Pools,
          Unlabelled])
sampl_size=ISZ=112
path = os.getcwd()
gnd_path = path + '/gt/'
img_path = path + '/sat/'
gnd_save_path = gnd_path + 'rotated/'
img_save_path = img_path + 'rotated/'
N_Cls=9
ISZ = 112
smooth = 1e-12

def create_image_from_one_hot(one_hot):
    Z_img=np.zeros((ISZ,ISZ,3))
    for i in range(0, one_hot[0].shape[0]):
        for j in range(0, one_hot[0].shape[1]):
            Z_img[i][j]=color_dict[np.argmax(one_hot[0][i][j])]
    return Z_img


json_file = open('model_112.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model_112.h5")
print("Loaded model from disk")



tiff = t.open('./sat/8.tif')                            #import a random image from given test-dataset
img = tiff.read_image()
tiff.close()

IP=img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

tiff = t.open('./gt/8.tif')
gnd = tiff.read_image()
tiff.close()

cou = 0
accuracy=0

gd = np.zeros(gnd.shape)
im = np.zeros(gnd.shape)
st = np.zeros(img.shape)

model.compile(optimizer=Nadam(lr=5e-4), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, categorical_accuracy, precision, recall])


for x in range(sampl_size, img.shape[0], sampl_size):
    for y in range(sampl_size, img.shape[1], sampl_size):
        
        im_ = img[x-sampl_size:x, y-sampl_size:y, :]#.astype('float32')
        gt_ = gnd[x-sampl_size:x, y-sampl_size:y, :]#.astype('float32')
        
        # count += 1
        
        im_ = im_.astype('float32').reshape((1, im_.shape[0], im_.shape[1], im_.shape[2]))

                
        output  = model.predict(im_)
        
        pred_im = create_image_from_one_hot(output)
        
        gd[x-sampl_size:x, y-sampl_size:y, :] = gt_
        im[x-sampl_size:x, y-sampl_size:y, :] = pred_im
        st[x-sampl_size:x, y-sampl_size:y, :] = img[x-sampl_size:x, y-sampl_size:y, :]

        GT_= gt_.astype('float32').reshape(1, gt_.shape[0], gt_.shape[1], gt_.shape[2])

        Z=np.zeros((1,ISZ,ISZ,9))
        for i in range(0, Z.shape[0]):
                for j in range(0, GT_.shape[1]):
                        for k in range(0, GT_.shape[2]):
                                count=0
                                for c in color_dict:
                                        if(np.array_equal(GT_[i][j][k],c)):
                                                Z[i][j][k]=to_categorical(count, 9)
                                        else:
                                                count=count+1

        score = model.evaluate(im_, Z, verbose=0)
        print(model.metrics_names[0])
        print(score[0]*100)
        cou+=1
        print(model.metrics_names[3])
        accuracy+=score[3]
        print(score[3]*100)
        

        plt.figure()
        plt.subplot(311)
        plt.imshow(gt_)
        plt.subplot(312)
        plt.imshow(img[x-sampl_size:x, y-sampl_size:y, :-1])
        plt.subplot(313)
        plt.imshow(pred_im)
        plt.show()


for x in range(sampl_size, img.shape[0], sampl_size):

    
        im_ = img[x-sampl_size:x, -sampl_size:, :]#.astype('float32')
        gt_ = gnd[x-sampl_size:x, -sampl_size:, :]#.astype('float32')

        im_ = im_.astype('float32').reshape((1, im_.shape[0], im_.shape[1], im_.shape[2]))
        
        
        output  = model.predict(im_)

        pred_im = create_image_from_one_hot(output)
      
        gd[x-sampl_size:x, -sampl_size:, :] = gt_
        im[x-sampl_size:x, -sampl_size:, :] = pred_im
        st[x-sampl_size:x, -sampl_size:, :] = img[x-sampl_size:x, -sampl_size:, :]

        GT_= gt_.astype('float32').reshape(1, gt_.shape[0], gt_.shape[1], gt_.shape[2])

        Z=np.zeros((1,ISZ,ISZ,9))
        for i in range(0, Z.shape[0]):
                for j in range(0, GT_.shape[1]):
                        for k in range(0, GT_.shape[2]):
                                count=0
                                for c in color_dict:
                                        if(np.array_equal(GT_[i][j][k],c)):
                                                Z[i][j][k]=to_categorical(count, 9)
                                        else:
                                                count=count+1

        score = model.evaluate(im_, Z, verbose=0)
        print(model.metrics_names[0])
        print(score[0]*100)

        cou+=1
        accuracy+=score[3]

        print(model.metrics_names[3])
        print(score[3]*100)

    
        plt.figure()
        plt.subplot(311)
        plt.imshow(gt_)
        plt.subplot(312)
        plt.imshow(img[x-sampl_size:x, -sampl_size:, :-1])
        plt.subplot(313)
        plt.imshow(pred_im)
        plt.show()
    
for y in range(sampl_size, img.shape[1], sampl_size):
    
    im_ = img[-sampl_size:, y-sampl_size:y, :]#.astype('float32')
    gt_ = gnd[-sampl_size:, y-sampl_size:y, :]#.astype('float32')

    im_ = im_.astype('float32').reshape((1, im_.shape[0], im_.shape[1], im_.shape[2]))
        
        
    output  = model.predict(im_)

    pred_im = create_image_from_one_hot(output)
      
    gd[-sampl_size:, y-sampl_size:y, :] = gt_
    im[-sampl_size:, y-sampl_size:y, :] = pred_im
    st[-sampl_size:, y-sampl_size:y, :] = img[-sampl_size:, y-sampl_size:y, :]

    GT_= gt_.astype('float32').reshape(1, gt_.shape[0], gt_.shape[1], gt_.shape[2])

    Z=np.zeros((1,ISZ,ISZ,9))
    for i in range(0, Z.shape[0]):

        for j in range(0, GT_.shape[1]):
                for k in range(0, GT_.shape[2]):
                       count=0
                       for c in color_dict:
                              if(np.array_equal(GT_[i][j][k],c)):
                                Z[i][j][k]=to_categorical(count, 9)
                              else:
                                count=count+1

    score = model.evaluate(im_, Z, verbose=0)
    print(model.metrics_names[0])
    print(score[0]*100)

    cou+=1
    accuracy+=score[3]

    print(model.metrics_names[3])
    print(score[3]*100)

    

    plt.figure()
    plt.subplot(311)
    plt.imshow(gt_)
    plt.subplot(312)
    plt.imshow(img[-sampl_size:, y-sampl_size:y, :-1])
    plt.subplot(313)
    plt.imshow(pred_im)
    plt.show()
        
im_ = img[-sampl_size:, -sampl_size:, :]#.astype('float32')
gt_ = gnd[-sampl_size:, -sampl_size:, :]#.astype('float32')

im_ = im_.astype('float32').reshape((1, im_.shape[0], im_.shape[1], im_.shape[2]))
        
        
output  = model.predict(im_)

pred_im = create_image_from_one_hot(output)
      
gd[-sampl_size:, -sampl_size:, :] = gt_
im[-sampl_size:, -sampl_size:, :] = pred_im
st[-sampl_size:, -sampl_size:, :] = img[-sampl_size:, -sampl_size:, :]

GT_= gt_.astype('float32').reshape(1, gt_.shape[0], gt_.shape[1], gt_.shape[2])
Z=np.zeros((1,ISZ,ISZ,9))
for i in range(0, Z.shape[0]):
        for j in range(0, GT_.shape[1]):
                for k in range(0, GT_.shape[2]):
                       count=0
                       for c in color_dict:
                        if(np.array_equal(GT_[i][j][k],c)):
                                Z[i][j][k]=to_categorical(count, 9)
                        else:
                                count=count+1

score = model.evaluate(im_, Z, verbose=0)
print(model.metrics_names[0])
print(score[0]*100)

cou+=1
accuracy+=score[3]

print(model.metrics_names[3])
print(score[3]*100)


plt.figure()
plt.subplot(311)
plt.imshow(gt_)
plt.subplot(312)
plt.imshow(img[-sampl_size:, -sampl_size:, :-1])
plt.subplot(313)
plt.imshow(pred_im)
plt.show()
      

print(cou)
print(accuracy)                 #overall accuracy
print(100*(accuracy/cou))

plt.figure()
plt.subplot(211)
plt.imshow(gd)
plt.subplot(212)
plt.imshow(im)
plt.show()

# tiff = t.open(gnd_path + 'tested_pred_8.tif', mode='w')                       #for saving predicted and original images
# tiff.write_image(im, None, True)
# tiff.close()

# tiff = t.open(gnd_path + 'tested_gndt_8.tif', mode='w')
# tiff.write_image(gd, None, True)
# tiff.close()

