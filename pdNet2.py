#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:09:03 2017

@author: iyouju
"""

import numpy as np
import tensorflow as tf
from PIL import Image
#import math
#from TFreader import *
import time
import os

FINE_TUNE = True
#FINE_TUNE = False

oldNetName = './npz2/simAlex5-4.data'
newNetName = './npz2/simAlex5-5.data'

IMG_H = 128
IMG_W = 64
IMG_CH = 3
LABEL_W = 2

NUM_EPOCHS = 1200
BATCH_SIZE = 40
DECAY_STEP = 200
SEED = 66478  # Set to None for random seed.

#DROPOUT = True
DROPOUT = False

checkpoint_path = os.path.join('./', 'model.ckpt')
checkpoint_dir = './'

def data_type():
    return tf.float32
    
def get_chunk(samples, labels, chunkSize):
    '''
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    '''
    if len(samples) != len(labels):
        raise Exception('Length of samples and labels must equal')
    stepStart = 0	# initial step
    i = 0
    while stepStart < len(samples):
        stepEnd = stepStart + chunkSize
        if stepEnd < len(samples):
            yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
            i += 1
        stepStart = stepEnd 
            
class pdNet:
    
    def __init__(self,train_file_name,test_file_name,batch_size,num_epochs=None):
        self.trainFileName = train_file_name
        self.testFileName = test_file_name
        self.train = True
        
        self.batchSize = batch_size
        self.numEpochs = num_epochs
        
        self.train_data_node = None
        self.train_labels_node = None
        
        self.graph = tf.Graph()

        self.defineGraph()
    def defineGraph(self):
    #        with self.graph.as_default():
            #----		net variables
            #    eval_data = tf.place
            with self.graph.as_default():
                #This is where training samples and labels are fed to the graph.
                #These placeholder nodes will be fed a batch of training data at each
                #training step using the (feed_dict) argument to the Run() call below.
                self.train_data_node = tf.placeholder(
                    data_type(),
                    shape=(BATCH_SIZE,IMG_H,IMG_W,IMG_CH),
                    name = 'train_data_node')
            
                self.train_labels_node = tf.placeholder(data_type(),shape=(BATCH_SIZE,LABEL_W),
                                                        name = 'train_labels_node')
                c1_w = tf.Variable(
                    tf.truncated_normal([11,11,IMG_CH,64],#5x5 filter depth 32.
                                        stddev=0.1,
                                        seed=SEED,dtype=data_type(),
                                        name = 'c1_w'))
                c1_b = tf.Variable(tf.zeros([64                                                                                                    ],dtype = data_type(),
                                        name = 'c1_b'))
                c2_w = tf.Variable(
                    tf.truncated_normal([5,5,64,192],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'c2_w'))
                c2_b = tf.Variable(tf.zeros([192],dtype = data_type(),
                                        name = 'c2_b'))
                c3_w = tf.Variable(tf.truncated_normal([3,3,192,384],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'c3_w'))
                c3_b = tf.Variable(tf.zeros([384],dtype=data_type(),
                                        name = 'c3_b'))
                                    
                c4_w = tf.Variable(tf.truncated_normal([3,3,384,256],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'c4_w'))
                c4_b = tf.Variable(tf.zeros([256],dtype=data_type(),
                                        name = 'c4_b'))                
                c5_w = tf.Variable(tf.truncated_normal([3,3,256,256],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'c4_w'))
                c5_b = tf.Variable(tf.zeros([256],dtype=data_type(),
                                        name = 'c4_b'))
                fc1_w = tf.Variable(
                    tf.truncated_normal([IMG_H//32*IMG_W//32*256,1640],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'fc1_w'))
                fc1_b = tf.Variable(tf.constant(0.1,shape=[1640],dtype=data_type(),
                                        name = 'fc1_b'))
                fc2_w = tf.Variable(
                    tf.truncated_normal([1640,LABEL_W],
                                        stddev=0.1,
                                        seed=SEED,dtype = data_type(),
                                        name = 'fc2_w'))
                fc2_b = tf.Variable(tf.constant(0.1,shape=[LABEL_W],dtype=data_type(),
                                        name = 'fc2_b'))
#                fc3_w = tf.Variable(
#                    tf.truncated_normal([512,LABEL_W],
#                                        stddev=0.1,
#                                        seed=SEED,dtype = data_type(),
#                                        name = 'fc2_w'))
#                fc3_b = tf.Variable(tf.constant(0.1,shape=[LABEL_W],dtype=data_type(),
#                                        name = 'fc2_b'))               
                #-----------------------------
#                tf.summary.histogram('c1_w',c1_w)
#                tf.summary.histogram('c1_b',c1_b)
#                tf.summary.histogram('c2_w',c2_w)
#                tf.summary.histogram('c2_b',c2_b)
#                tf.summary.histogram('c3_w',c1_w)
#                tf.summary.histogram('c3_b',c3_b)
#                tf.summary.histogram('c4_w',c4_w)
#                tf.summary.histogram('c4_b',c4_b)
#                tf.summary.histogram('fc1_w',fc1_w)
#                tf.summary.histogram('fc1_b',fc1_b)
                tf.summary.histogram('fc2_w',fc2_w)
                tf.summary.histogram('fc2_b',fc2_b)
                #---------------------	end net variables
                #
                def model(data,train=False):
                    # shape matches the data layout:[image index,y,x,depth].
                    data = data*1/255-0.5
                    c1 = tf.nn.conv2d(data,
                                      c1_w,
                                      strides=[1,4,4,1],
                                        padding = 'SAME')
                    # Bias and rectified linear non-linearity.
                    sig1 = tf.nn.sigmoid(tf.nn.bias_add(c1,c1_b))
#                    relu1 = tf.nn.relu(tf.nn.bias_add(c1,c1_b))
                    # Max pooling
        
                    pool1 = tf.nn.max_pool(sig1,
                                          ksize=[1,3,3,1],
                                            strides=[1,2,2,1],
                                            padding='SAME')
                    c2 = tf.nn.conv2d(pool1,
                                      c2_w,
                                      strides=[1,1,1,1],
                                        padding='SAME')
                    sig2 = tf.nn.sigmoid(tf.nn.bias_add(c2,c2_b))
                    pool2 = tf.nn.max_pool(sig2,
                                           ksize=[1,5,5,1],
                                            strides=[1,2,2,1],
                                            padding='SAME')
                    c3 = tf.nn.conv2d(pool2,
                                      c3_w,
                                      strides=[1,1,1,1],
                                      padding = 'SAME')
                    sig3 = tf.nn.sigmoid(tf.nn.bias_add(c3,c3_b))
                    c4 = tf.nn.conv2d(sig3,
                                      c4_w,
                                      strides=[1,1,1,1],
                                      padding = 'SAME')
                    sig4 = tf.nn.sigmoid(tf.nn.bias_add(c4,c4_b))
#                    pool4 = tf.nn.max_pool(sig4,
#                                            ksize=[1,3,3,1],
#                                            strides=[1,2,2,1],
#                                            padding='SAME'
#                                            )
                    c5 = tf.nn.conv2d(sig4,
                                      c5_w,
                                      strides=[1,1,1,1],
                                      padding = 'SAME')
                    sig5 = tf.nn.sigmoid(tf.nn.bias_add(c5,c5_b))
                    pool5 = tf.nn.max_pool(sig5,
                                           ksize=[1,3,3,1],
                                           strides=[1,2,2,1],
                                           padding='SAME')
                    cnnOut = pool5
                    # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
                    cnnOutShape = cnnOut.get_shape().as_list()
                    reshape = tf.reshape(
                                        cnnOut,
                                        [cnnOutShape[0],cnnOutShape[1]*cnnOutShape[2]*cnnOutShape[3]])
                    # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
                    fc1 = tf.nn.sigmoid(tf.matmul(reshape,fc1_w) + fc1_b)
                    # Add a 50% dropout during training training only.
                    # Dropout also scales activations such that no rescaling is needed at evaluation time
                    if DROPOUT:
                        fc1 = tf.nn.dropout(fc1,0.5,seed=SEED)
                    fc2n = tf.nn.bias_add(tf.matmul(fc1,fc2_w), fc2_b)
                    fc2 = tf.nn.sigmoid(fc2n)
                    
#                    tf.summary.histogram('fc1',fc1)
#                    tf.summary.histogram('fc2',fc2)
                    
                    return fc2
                #--------------------end model
    
                # Training computation: logits + cross-entropy loss
                logits = model(self.train_data_node,self.train)
                self.loss1 = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.train_labels_node))
                # L2 regularization for the fully connected parameters.
                regularizers = (tf.nn.l2_loss(fc1_w) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc2_b))

                # Add the regularization term to the loss.
    #            
    #            if  tf.is_nan(regularizers) is not None:
    #                self.loss = self.loss1*(1 + tf.nn.tanh(1e-2*regularizers))
    #            else:
    #                self.loss = self.loss1
#                loss = self.loss1 + 0.5*tf.nn.tanh(1e-5*regularizers)
    #            regularizers = tf.nn.tanh(regularizers)
#                self.loss = self.loss1 + regularizers
                loss = self.loss1
                
#                tf.summary.scalar('loss1',self.loss1)
                # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
                batch = tf.Variable(0,dtype=data_type())
                # Decay once per epoch, using an exponential schedule starting at 0.01
                self.learningRate = tf.train.exponential_decay(
                    0.01,               # Base learning rate.
                    batch, # Current index into the dataset
                    DECAY_STEP,         # Decay step.
                    0.97,               # Decay rate.
                    staircase=True)
#                learningRate = tf.train.exponential_decay(
#                    0.03,               # Base learning rate.
#                    batch, # Current index into the dataset
#                    DECAY_STEP,         # Decay step.
#                    0.9,               # Decay rate.
#                    staircase=True)
#                self.learningRate = learningRate
                learningRate = 0.0003
                self.optimizer = tf.train.MomentumOptimizer(learningRate,0.9).minimize(loss,global_step=batch)
        
                # Predictions for the current training minibatch
                self.trainPrediction = tf.nn.softmax(logits)
#                self.trainPrediction = (logits)
                
                #-------------------------------------------------
                tf.summary.scalar('regularizers',regularizers)
                tf.summary.scalar('loss1',self.loss1)
#                tf.summary.scalar('loss',loss)
                tf.summary.scalar('learningRate',self.learningRate)
                
                #----------------------------------------------------
                
    #        self.merge = tf.merge_v2_checkpoints()
    #        self.merge = tf.merge_all_summaries()
        #------------   END def defineGraph()
    def error_rate(self,predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        numP = len(predictions)
        numL = len(labels)
        if not numP==numL:
            print("lenPred != lenLabels!")
            return -1
        pmax = np.argmax(predictions, 1)
        lmax = np.argmax(labels, 1)
        s = np.sum(pmax==lmax)
        rate = 100.0 * (1.0 - np.float32(s)/np.float32(numP))
        return rate
        
    def input(self):
        if self.train:
            dat = np.load(self.trainFileName)
        else:
            dat = np.load(self.testFileName)
    
        imgs = dat['img']
        labs = dat['lab']
        print('Input data.')
        return imgs,labs
    
    def trainNet(self):
        self.train = True
        self.session = tf.Session(graph=self.graph)
        with self.session as session:
            
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("output/",self.graph)
            
            if FINE_TUNE:
                try:
                    saver = tf.train.Saver(tf.global_variables()) # 'Saver' misnomer! Better: Persister!
                except:pass
                saver.restore(session, oldNetName)
            else:
                tf.local_variables_initializer().run(session=session) # epoch计数变量是local variable
                tf.global_variables_initializer().run(session=session)
            img,lab = self.input()
            sumI = 0
            for epoch in range(self.numEpochs):
                startTime = time.time()
                predict = []
                goldLabel = []
                for step,imgBatch,labBatch in get_chunk(img,lab,self.batchSize):
                    feed_dict = {self.train_data_node: imgBatch,
                                 self.train_labels_node: labBatch}
                    _,loss1,lr,sumRe,prediction = session.run([self.optimizer,
                                                               self.loss1,
                                                               self.learningRate,
                                                               merged,
                                                               self.trainPrediction],
                                                                feed_dict = feed_dict)
                    if not step%30:
                        writer.add_summary(sumRe,sumI)
                    sumI += 1
            
                    if not step:
                        predict = prediction
                        goldLabel = labBatch
                    else:
                        predict = np.vstack((predict,prediction))
                        goldLabel = np.vstack((goldLabel,labBatch))
                elapsed_time = time.time() - startTime    
                err = self.error_rate(predict,goldLabel)
                print('epoch:%d,\tepochTime:%f s,\terr:%.2f\tloss1:%.4f\tlr:%f' %(epoch,elapsed_time,err,loss1,lr))
                
                if not epoch%20:
                    saver = tf.train.Saver(tf.global_variables())
                    saver.save(session, newNetName)        
                    print('Net has been trained and saved.')
                    #   Validation
                    predict = []
                    goldLabel = []
                    imgValid = img[0:700]
                    labValid = lab[0:700]
                    for step,imgBatch,labBatch in get_chunk(imgValid,labValid,self.batchSize):
                        feed_dict = {self.train_data_node: imgBatch,
                                     self.train_labels_node: labBatch}
                        _,sumRe,prediction = session.run([self.optimizer,
                                                                     merged,
                                                                 self.trainPrediction],
                                                                 feed_dict = feed_dict)
                        if not step:
                            predict = prediction
                            goldLabel = labBatch
                        else:
                            predict = np.vstack((predict,prediction))
                            goldLabel = np.vstack((goldLabel,labBatch))
                    err = self.error_rate(predict,goldLabel)
                    print('validation_err_rate: %f' %err)
                    #   evaluation
                    self.evalu()
                    
#                    if DROPOUT:
#                        DROPOUT = False
#                    else:
#                        DROPOUT = True
                    
            
    def evalu(self):
        self.train = False
        self.session = tf.Session(graph=self.graph)
        with self.session as session:
            try:
                saver = tf.train.Saver(tf.global_variables()) # 'Saver' misnomer! Better: Persister!
            except:pass
            saver.restore(session, newNetName)
            
            img,lab = self.input()
            predict = []
            goldLabel = []
            for step,imgBatch,labBatch in get_chunk(img,lab,self.batchSize):
#                startTime = time.time()
                feed_dict = {self.train_data_node: imgBatch,
                             self.train_labels_node: labBatch}
                re = session.run(self.trainPrediction,
                                 feed_dict = feed_dict)
                
                if not step:
                    predict = re
                    goldLabel = labBatch
                else:
                    predict = np.vstack((predict,re))
                    goldLabel = np.vstack((goldLabel,labBatch))
                    
            err = self.error_rate(predict,goldLabel)
            print('evalu_err_rate: %f' %err)

        print("Evalu is over.") 
if __name__ == '__main__':
#    net = pdNet("trainData_64x128.tfrecords","trainData_64x128.tfrecords",BATCH_SIZE,num_epochs=NUM_EPOCHS)
    net = pdNet("trainData_128x64.npz","testData_128x64.npz",BATCH_SIZE,num_epochs=NUM_EPOCHS)
#    net.trainNet()
    net.evalu()
    del net
    print("Procession is over.")