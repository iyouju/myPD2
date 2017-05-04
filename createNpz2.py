#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:53:14 2017

@author: iyouju
"""

import os
import random
import numpy as np
#from scipy import io #io.savemat

#import tensorflow as tf
from PIL import Image

#-------------------------------
IMG_H = 128
IMG_W = 64
IMG_CH = 3
LABEL_W = 2
NEGLABEL = [1,0]
POSLABEL = [0,1]
PROPORTION_TRAIN_DATA = 0.8
negPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/neg/'
posPath = '/home/iyouju/pythonPro/DATASETS/INRIAPerson/train_64x128_H96/pos/'
#--------------------------------

#
def addImg2File(fileName,pathList,labelList):
    negUnst = 0
    posUnst = 0
    l = len(pathList)
    
#    imgSet = np.ndarray([l,IMG_H,IMG_W,IMG_CH])
#    labset = np.ndarray([l,LABEL_W])
    imgSet = []
    labSet = []
    imgNum = 0
    for i in range(l):
        path = pathList[i]
        index = labelList[i]
        print(path)
        imgRaw = Image.open(path)
                    
        if index[1]:
            temp = imgRaw.resize((IMG_W, IMG_H),Image.BILINEAR)
            img_arr = np.asarray(np.uint8(temp))
            sh = img_arr.shape
            imgSet.append(img_arr[:,:,0:3])
            labSet.append(index)
            imgNum += 1
        else:
            for i in range(3):
                img_arr = np.asarray(np.uint8(imgRaw))
                sh = img_arr.shape
                h = random.randint(0,sh[0]-IMG_H)
                w = random.randint(0,sh[1]-IMG_W)
                imgSet.append(img_arr[h:h+IMG_H,w:w+IMG_W,0:3])
                labSet.append(index)
                imgNum += 1
    infor = {'IMG_NUM':imgNum,
            'IMG_H':IMG_H,
            'IMG_W':IMG_W,
            'IMG_CH':IMG_CH,
            'LABEL_W':LABEL_W}
    np.savez(fileName,img=imgSet,lab=labSet,info=infor)
#    np.savez(fileName,img=imgSet,lab=labSet)
    print('posUnst: %d,\tnegUnst: %d' %(posUnst,negUnst))
    
    return imgNum
#    print('imgNum:%d' %imgNum)

#create tfrecords
def createNpz():
    negNameList = []
    negNameList = os.listdir(negPath)
    posNameList = []
    posNameList = os.listdir(posPath)

    imgPath = []
    imgLabel = []
    for img in negNameList:
        imgPath.append(negPath+img)
        imgLabel.append(NEGLABEL)
    for img in posNameList:
        imgPath.append(posPath+img)
        imgLabel.append(POSLABEL)
    numTotal = len(imgPath)
    numTrainData = np.floor(numTotal * PROPORTION_TRAIN_DATA)
#    numTestData = numTotal - numTrainData
    
    index = range(numTotal)
    random.shuffle(index)
    pathListTrain = []
    labelListTrain = []
    pathListTest = []
    labelListTest = []
    
    cnt = 0
    for i in index:
        if cnt < numTrainData:
            pathListTrain.append(imgPath[i])
            labelListTrain.append(imgLabel[i])
        else:
            pathListTest.append(imgPath[i])
            labelListTest.append(imgLabel[i])
        cnt += 1
     
    trainNum = addImg2File("trainData_128x64",pathListTrain,labelListTrain)
    testNum = addImg2File("testData_128x64",pathListTest,labelListTest)
    print('trianNum:%d,testNum:%d' %(trainNum,testNum))
    print('negLen:%d\tposLen:%d' %(len(negNameList),len(posNameList)))

#-------------------------------

if __name__ == '__main__':
    createNpz()
    print('createTFrecords is over.')
