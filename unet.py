from __future__ import print_function
import numpy as np
import pandas as pd
import os
import random
import glob
import cv2
from libtiff import TIFF as t
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Cropping2D, concatenate, ZeroPadding2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Nadam
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy, precision, recall
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
import collections

# In[3]:
random.seed(16)  #16, 42 is a good start

N_Cls=9
ISZ = 112
smooth = 1e-12
N_imgs=3300

path = os.getcwd()
path = os.getcwd()
gnd_path = path + '/gt/'
img_path = path + '/sat/'
gnd_save_path = gnd_path + 'rotated/'
img_save_path = img_path + 'rotated/'

IP=np.random.randint(0,255,(N_imgs, ISZ, ISZ, 4))
OP=np.random.randint(0,255,(N_imgs, ISZ, ISZ,3))


def dataset_rot():
    count = 0
    img_count_I=0
    img_count_O=0

    final_size  = ISZ/2
    sample_size = 2*final_size #give some safety margin over sqrt(2)
    num_rotations = 3 #one is 0 rotation and then clk, cclk
    for image_path in glob.glob(img_path + '*.tif'):

        name = image_path.split('/')[-1]

        tiff = t.open(image_path)
        img_sat = tiff.read_image()
        tiff.close()

        tiff = t.open(gnd_path + name)
        img_gt = tiff.read_image()
        tiff.close()

        rows, cols, _ = img_sat.shape

        for x in range(sample_size, img_sat.shape[0], sample_size):
            for y in range(sample_size, img_sat.shape[1], sample_size):
                temp_sat = img_sat[x-sample_size:x,y-sample_size:y,:]
                temp_gt  = img_gt[x-sample_size:x,y-sample_size:y,:]

                rows, cols, _ = temp_sat.shape
                cx = rows/2
                cy = cols/2
                im_ = temp_gt[cx-final_size:cx+final_size, cy-final_size:cy+final_size, :]
                
                OP[img_count_O, :, :, :] = im_
                img_count_O+=1

                im_ = temp_sat[cx-final_size:cx+final_size, cy-final_size:cy+final_size, :]                    
                IP[img_count_I,:,:,:] = im_
                img_count_I+=1

                for j in range((num_rotations-1)/2):
                    angle_cclk = 90
                    angle_clk  = -90
                    M_cclk = cv2.getRotationMatrix2D((cols/2, rows/2), angle_cclk, 1)               #rotating image as part of data augmentation
                    M_clk  = cv2.getRotationMatrix2D((cols/2, rows/2), angle_clk, 1)
                    temp_sat_cclk = cv2.warpAffine(temp_sat, M_cclk, (cols, rows))
                    temp_sat_clk  = cv2.warpAffine(temp_sat, M_clk, (cols, rows))
                    temp_gt_cclk = cv2.warpAffine(temp_gt, M_cclk, (cols, rows))
                    temp_gt_clk  = cv2.warpAffine(temp_gt, M_clk, (cols, rows))

                    cx_cc, cy_cc, _ = temp_sat_cclk.shape
                    cx_c, cy_c, _   = temp_sat_clk.shape
                    cx_cc /= 2
                    cy_cc /= 2
                    cx_c  /= 2
                    cy_c  /= 2

                    sat_64_cclk = temp_sat_cclk[cx_cc-final_size:cx_cc+final_size, cy_cc-final_size:cy_cc+final_size, :]
                    gt_64_cclk  = temp_gt_cclk[cx_cc-final_size:cx_cc+final_size, cy_cc-final_size:cy_cc+final_size, :]
                    sat_64_clk  = temp_sat_clk[cx_c-final_size:cx_c+final_size, cy_c-final_size:cy_c+final_size, :]
                    gt_64_clk   = temp_gt_clk[cx_c-final_size:cx_c+final_size, cy_c-final_size:cy_c+final_size, :]

                    OP[img_count_O,:,:,:] = gt_64_cclk
                    img_count_O+=1

                    OP[img_count_O,:,:,:] = gt_64_clk
                    img_count_O+=1

                    IP[img_count_I,:,:,:] = sat_64_cclk
                    img_count_I+=1

                    IP[img_count_I,:,:,:] = sat_64_clk
                    img_count_I+=1

                count += num_rotations
    print(img_count_O)
    print(img_count_I)

# In[4]:
dataset_rot()


def jaccard_coef(y_true, y_pred):
    #custom metric for evaluation
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):

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

# In[5]:
def get_crop_shape(target, refer):
        # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def create_model(img_shape, num_class):

    concat_axis = 3
    inputs = Input(shape = img_shape)

    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(norm1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(conv2)
    drop1 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv3)
    norm2 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(norm2)

    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv4)
    drop2 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', border_mode='same')(conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    drop3 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(drop3)
    conv6 = Conv2D(256, (3, 3), activation='relu', border_mode='same')(conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    norm3 = BatchNormalization()(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(norm3)
    conv7 = Conv2D(128, (3, 3), activation='relu', border_mode='same')(conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    drop4 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', border_mode='same')(drop4)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    norm4 = BatchNormalization()(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(norm4)
    conv9 = Conv2D(32, (3, 3), activation='relu', border_mode='same')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = Conv2D(num_class, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Nadam(lr=5e-4), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, categorical_accuracy, precision, recall])

    return model
model = create_model((ISZ,ISZ,4),9)



# In[9]:


Roads = [0,0,0]
Water = [0, 0, 150]
Trees = [0, 125, 0]                     #colors of various given classes
Grass = [0, 255, 0]
Buildings = [100, 100, 100]
Bare_soil = [150, 80, 0]
Rails = [255, 255, 0]
Pools = [150, 150, 255]
Unlabelled = [255, 255, 255]




color_dict=np.array([Roads,
          Water,                    #array of various classes
          Trees,
          Grass,
          Buildings,
          Bare_soil,
          Rails,
          Pools,
          Unlabelled])







# In[15]:

Z=np.zeros((N_imgs,ISZ,ISZ,9))
for i in range(0, Z.shape[0]):
   for j in range(0, OP.shape[1]):                          #one-hot encoding of the input image pixels in accordance with the order in color_dict
       for k in range(0, OP.shape[2]):
           count=0
           for c in color_dict:
               if(np.array_equal(OP[i][j][k],c)):
                   Z[i][j][k]=to_categorical(count, 9)
               else:
                   count=count+1

for i in range(1):
    model.fit(IP,Z,batch_size=15, epochs=50, shuffle=True, verbose=2, validation_split=0.2)


