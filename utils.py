from tensorflow.keras import backend as K
import os
import numpy as np
import random
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout,BatchNormalization,ZeroPadding2D,add, Flatten,Activation,AveragePooling2D,Dense
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.layers import Lambda
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model,to_categorical
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from sklearn.metrics import confusion_matrix
import pandas as pd


def OverallAccuracy(confusionMatrix):  
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def IntersectionOverUnion(confusionMatrix):  
    #  返回交并比IoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    #  返回平均交并比mIoU
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    #  返回频权交并比FWIoU
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def val_cal(label_arr,gt_arr,out_file,if_show=True):
    y_true=gt_arr.reshape((gt_arr.shape[0]*gt_arr.shape[1]*gt_arr.shape[2],1))
    y_predict=label_arr.reshape((label_arr.shape[0]*label_arr.shape[1]*label_arr.shape[2],1))
    # a=classification_report(y_true,y_predict)
    cnf_matrix=confusion_matrix(y_true, y_predict)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    iou=TP/(FP+FN+TP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP+FP+FN)
    # classification_report(y_true,y_predict)
    oa = OverallAccuracy(cnf_matrix)
    FWIOU = Frequency_Weighted_Intersection_over_Union(cnf_matrix)
    mIOU = MeanIntersectionOverUnion(cnf_matrix)
    data=np.ones(precision.shape)
    oa=data*oa
    mIOU=data*mIOU
    FWIOU=data*FWIOU
       
    result=np.column_stack((accuracy,precision,recall,fscore,iou,oa,mIOU,FWIOU))
    mean=np.mean(result,axis=0)
    result=np.vstack((result,mean.transpose()))
    name=['accuracy','precision','recall','F1-score','iou','oa','miou','fwiou']
    df2 = pd.DataFrame((result))
    df2.index=['crop','tree','grass','water','building','others','mean']#耕地=1，林地=2，草地=3，水域=4，居民地=5，未利用=6
    df2.columns=name
    # out_file='./csv.csv'
    df2.to_csv(out_file, index=True, encoding="utf_8_sig")
    if if_show:
        print(df2)
    return result
    
def label_hot(label,n_label=1):
    listlabel=[]
    for i in label:
        mask=i.flatten()
        mask=to_categorical(mask, num_classes=n_label)
        listlabel.append(mask)
    msk=np.asarray(listlabel,dtype='uint16')
    msk=msk.reshape((label.shape[0],label.shape[1],label.shape[2],n_label))
#     print(msk.shape)
    return msk
def get_normalized_patches(data,label,n_label=1):
#     data = get_all_patches()
#     data = np.load(Dir + '\output\data_pos_%d_%d_class%d.npy' % (Patch_size, N_split, Class_Type))
    img = data/255.0
    # img=np.asarray(img,dtype='uint16')
    msk = label_hot(label,n_label)
    print(img.shape,msk.shape)
#     combin=np.expand_dims(combin, axis=3)
#     img=np.concatenate((img,combin),axis=-1)
    return img,msk
def checkmsk_plot_func(label,n_label):
    fig=plt.figure(figsize=(25,5))
    for i in range(n_label-1):
        plt.subplot(2,10,i+1)
        plt.imshow((label[:,:,i+1]),cmap="gray")
    plt.show()
 
def datagen(xtrain,ytrain,bs):
    seed = np.random.randint(0, 100)+10
    data_gen_args = dict(
                     # rescale=1/10000.0,
#                      featurewise_center=True,
#                      featurewise_std_normalization=True,
                     rotation_range=0.0,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=False,
                     vertical_flip=False,
                     zoom_range = 0.1,
                     channel_shift_range=0,
                     preprocessing_function=None)
    label_gen_args = dict(
#                      rescale=1/255.0,
                     rotation_range=0.0,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=False,
                     vertical_flip=False,
                     zoom_range = 0.1,
                     channel_shift_range=0)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**label_gen_args)
    # fits the model on batches with real-time data augmentation:
    dataiter=image_datagen.flow(xtrain,seed=seed, batch_size=bs)
    labeliter=mask_datagen.flow(ytrain,seed=seed, batch_size=bs)
    train_generator = zip(dataiter, labeliter)
    for (img,mask) in train_generator:
#         img,mask = randomColor_processing(img,mask)
        # img=img/255.0
        # mask=mask/255.0
        yield (img,mask)
def plot_func(data,label):
    fig=plt.figure(figsize=(25,5))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.imshow(Image.fromarray(np.uint8((data[i,:,:,0:3])*255)))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):
        plt.subplot(2,10,i+1)
        plt.imshow((label[i,:,:]),cmap="gray")
    plt.show()
def val_plot_func(data,label,yval):
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow(Image.fromarray(np.uint8((data[i,:,:,0:3])*255)))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow((label[i,:,:]))
    plt.show()
    fig=plt.figure(figsize=(25,5))
    for i in range(20):   
        plt.subplot(2,10,i+1)
        plt.imshow((yval[i,:,:]))
    plt.show()