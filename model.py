#!/usr/bin/python
# model.py
# CS 230 - Predicting Flight Delays with Deep Learning - {ethanchi, jilingt}@stanford.edu
# Implements a model using CNN preprocessing + LSTM sequential model inspired by LSTNet (Lai, 2018).

import numpy as np
import tensorflow as tf
import logging
import sys
import pandas as pd
import time

# constants
USE_CNN = True                  # set to False to use dense preprocessing layers instead
NUM_HIDDEN_PRP_LAYERS = 3       # only used if USE_CNN = false, for dense preprocessing layers
NUM_HIDDEN_RNN = 6              # the # of features yielded from LSTM hidden layer
NUM_HIDDEN_NN = 16              # layer 1
NUM_HIDDEN_LAYERS = 5           # the # of hidden layers
SEQUENCE_LENGTH = 15            # the length of sequences considered by our RNN
PRINT_RATE = 20                 # how often to print debug messages
BATCH_SIZE = 5000               # batch size for minibatch
LEARNING_RATE = 0.0005          # the learning rate for optimization
NUM_CONVNET_FILTERS = 6         # the # of convnet filter outputs
NUM_EPOCHS = 500                # the # of epochs to train for
FILTER_WIDTH = 6                # the width of the CNN preprocessing filter
LARGE_SIZE = 10**8              # a very large size used for scratch space in dataset processing, should > # of elems
REGULARIZATION = 0.003          # the degree of L1 regularization used
CUTOFF = 0.5                    # used for statistics - the cutoff to regard an output as a 1 (vs. 0)
EPSILON = 10**-7                # a small constant used for numerical stability

#configuration
np.set_printoptions(threshold=sys.maxsize, precision=3)

# Loads data from a filename given by preprocessing4.py.
# Returns the data we work off (i.e. weather data, day/time, etc.) and delays.
def loadData(filename):
    data = pd.read_csv(filename)
    delays = data.loc[:, data.columns == 'ARR_DELAY']
    arrCols = list(filter(lambda x: x.startswith('a_'), data.columns))
    arrData = data[arrCols]
    arrData = arrData.drop('a_join_time', axis=1)
    depCols = list(filter(lambda x: not x.startswith('a_'), data.columns))
    depData = data
    depData = depData.drop([col for col in ('ARR_DELAY', 'year', 'Unnamed: 0', 'a_join_time') if col in depData.columns], axis=1)
    depData = depData.drop(['OP_CARRIER_FL_NUM', 'wxcodes', 'join_time', 'Number', 'id', 'ORIGIN_AIRPORT_ID', 'DEP_DELAY'], axis=1)
    return [np.copy(x.values) for x in (depData, arrData, delays)]

# Building sequences generator
def datasetGenerator(numArrDataFeatures, numDepDataFeatures, useBatch=True, shuffle=True):
    arrDataPlc = tf.placeholder(tf.float32, (None, numArrDataFeatures), name='arrData')
    delaysPlc = tf.placeholder(tf.float32, (None, 1), name='delays')
    plc = tf.concat((arrDataPlc, delaysPlc), axis=1) 
    
    #dataset management
    
    #the previously departed flights' data
    ds = tf.data.Dataset.from_tensor_slices(plc)
    ds = ds.window(SEQUENCE_LENGTH, drop_remainder=True, shift=1)
    ds = ds.flat_map(lambda x: x.batch(SEQUENCE_LENGTH))
    
    # the data for the current flight
    depInfoPlc = tf.placeholder(tf.float32, (None, numDepDataFeatures), name='depData')
    depInfoDS = tf.data.Dataset.from_tensor_slices(depInfoPlc)
    depInfoDS = depInfoDS.skip(SEQUENCE_LENGTH)  # we can only start predicting after SEQUENCE_LENGTH elements
    
    # the ground truth
    YPlc = tf.placeholder(tf.float32, (None, 1), name='Y')
    YDS = tf.data.Dataset.from_tensor_slices(YPlc)
    YDS = YDS.skip(SEQUENCE_LENGTH)
    
    # combning into one large dataset, then batching
    ds = tf.data.Dataset.zip((ds, depInfoDS, YDS))
    ds = ds.batch(BATCH_SIZE if useBatch else LARGE_SIZE, drop_remainder=useBatch)
    if shuffle: ds = ds.shuffle(10**8, reshuffle_each_iteration=True)
    return arrDataPlc, delaysPlc, depInfoPlc, YPlc, ds

# Creates a tensorflow network using LSTNet.
def createModelLSTNet(arrData, depData, Y):

    useDropout = tf.placeholder(tf.bool)

    # step 1: CNN
    if USE_CNN:
        arrDataCNN = tf.expand_dims(arrData, -1)
        arrDataCNN = tf.reshape(arrDataCNN, (-1, arrData.shape[2], SEQUENCE_LENGTH, 1))
        numVariables = int(arrData.shape[2])
        arrDataConvoluted = tf.keras.layers.Conv2D(NUM_CONVNET_FILTERS, (numVariables, SEQUENCE_LENGTH), kernel_regularizer=tf.contrib.layers.l1_regularizer(REGULARIZATION))(arrDataCNN)
        arrDataConvoluted = tf.reshape(arrDataConvoluted, (-1, arrDataConvoluted.shape[-1], 1))
    else:
        for i in range(NUM_HIDDEN_PRP_LAYERS):
            layer = tf.layers.dense(arrData if i == 0 else prpNNOutputs[-1], 8, activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l1_regularizer(REGULARIZATION))
            dropout = tf.layers.dropout(layer, training=useDropout)
            prpNNOutputs.append(dropout)
        arrDataConvoluted = prpNNOutputs[-1]

    # step 2: RNN
    lstmCell = tf.contrib.rnn.LSTMBlockCell(NUM_HIDDEN_RNN, forget_bias=1.0, name='LSTMcell', use_peephole=True)
    rnnOutputs, _ = tf.nn.dynamic_rnn(lstmCell, inputs=arrDataConvoluted, dtype=np.float32)

    # step 3: dense layer
    combined = tf.concat((rnnOutputs[:, -1, :], depData), axis=1)
    nnOutputs = []
    for i in range(NUM_HIDDEN_LAYERS):
        numHiddenUnits = int(50 - abs(NUM_HIDDEN_LAYERS - i) * 8) # so that the number of hidden units decreases linearly
        layer = tf.layers.dense(combined if i == 0 else nnOutputs[-1], numHiddenUnits, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l1_regularizer(REGULARIZATION))
        dropout = tf.layers.dropout(layer, training=useDropout) # rate=(0.5 - abs(NUM_HIDDEN_LAYERS - i) * 0.4))
        nnOutputs.append(dropout)

    # step 4: prediction
    prediction = tf.layers.dense(nnOutputs[-1], 1, activation=tf.nn.sigmoid,
    kernel_initializer=tf.contrib.layers.xavier_initializer(),
    kernel_regularizer=tf.contrib.layers.l1_regularizer(REGULARIZATION))
    with tf.device('/cpu:0'):  # log loss doesn't work on our GPU
        loss = tf.losses.log_loss(prediction, Y, weights=(Y + 0.6)) + tf.losses.get_regularization_loss()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE, name='optimizer') 
    minimize = optimizer.minimize(loss)
    return minimize, prediction, Y, loss, useDropout

# Calculates statistics for a given run.
def calculateStatistics(arr, f=sys.stdout):
    arr = (arr > CUTOFF) # the cutoff
    total = arr.shape[0]

    # statistics
    positives = np.where(arr[:,1] == True)[0].shape[0]
    negatives = np.where(arr[:,1] == False)[0].shape[0]
    truePositives = np.where(np.all((arr[:,0]==True, arr[:,1] == True), axis=0))[0].shape[0]
    trueNegatives = np.where(np.all((arr[:,0]==False, arr[:,1] == False), axis=0))[0].shape[0]
    falsePositives = np.where(np.all((arr[:,0]==True, arr[:,1] == False), axis=0))[0].shape[0]
    falseNegatives =np.where(np.all((arr[:,0]==False, arr[:,1] == True), axis=0))[0].shape[0] 
    
    precision = truePositives/(truePositives + falsePositives + EPSILON) #we use EPSILON for numerical stability
    recall = truePositives/(truePositives + falseNegatives + EPSILON)
    accuracy = (truePositives + trueNegatives)/total
    
    f1 = 2 * (precision * recall) / (precision + recall + EPSILON)
    print("Batch size: {}\nRegularization: {}\nLearning rate: {}".format(BATCH_SIZE, REGULARIZATION, LEARNING_RATE), file=f)
    print("Total:", total, file=f)
    print("Positives:", positives, file=f)
    print("Negatives:", negatives, file=f)
    print("True positives: {} ({:.2%})".format(truePositives, truePositives/total), file=f)
    print("False positives: {} ({:.2%})".format(falsePositives, falsePositives/total), file=f)
    print("False negatives: {} ({:.2%})".format(falseNegatives, falseNegatives/total), file=f)
    print("Accuracy: {}".format(accuracy), file=f)
    print("Precision:", precision, file=f)
    print("Recall:", recall, file=f)
    print("F1 score", f1, file=f)
    return precision, recall, f1

# Splits the data into train, dev, test sets.
def split(x, kind):
    numElems = int(x.shape[0])
    eight = int(numElems * 0.8)
    nine = int(numElems * 0.9)
    if kind == "TRAIN":
        return x[:eight]
    elif kind == "DEV":
        return x[eight:nine]
    elif kind == "TEST":
        return x[nine:] 

def main():
    tf.reset_default_graph()
    dataOutput = loadData(sys.argv[1])
    depData, arrData, delays = [split(x, "TRAIN") for x in dataOutput]
    testDepData, testArrData, testDelays = [split(x, "TEST") for x in dataOutput]
    numRows = depData.shape[0]
    arrDataPlc, arrDelaysPlc, depDataPlc, depDataDelays, arrDataDS = datasetGenerator(arrData.shape[1], depData.shape[1])
    arrDataUnb, arrDelaysUnb, depDataUnb, depDataDelaysUnb, arrDataUnbDS = datasetGenerator(arrData.shape[1], depData.shape[1], useBatch=False, shuffle=False)
  
    numArrFeatures = arrData.shape[1]
    numDepFeatures = depData.shape[1]

    iter = arrDataDS.make_initializable_iterator()
    iterNext = iter.get_next()
    iterUnb = arrDataUnbDS.make_initializable_iterator()
    iterUnbNext = iterUnb.get_next()
    numSequences = (numRows - SEQUENCE_LENGTH + 1)
    
    arrDataPlaceholder = tf.placeholder(tf.float32, (None, SEQUENCE_LENGTH, numArrFeatures+1))
    depDataPlaceholder = tf.placeholder(tf.float32, (None, numDepFeatures))
    yPlaceholder = tf.placeholder(tf.float32, (None, 1))
    with tf.device('/gpu:0'):
        minimize, prediction, yData, loss, useDropout = createModelLSTNet(arrDataPlaceholder, depDataPlaceholder, yPlaceholder)
        
    # initialize the cache for model input
        numTimes = numSequences // BATCH_SIZE
        datastore = (np.zeros((numTimes, BATCH_SIZE, SEQUENCE_LENGTH, numArrFeatures+1)),
                     np.zeros((numTimes, BATCH_SIZE, numDepFeatures)),
                     np.zeros((numTimes, BATCH_SIZE, 1)))
        
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        sess.run(iter.initializer, feed_dict={arrDataPlc:arrData, arrDelaysPlc: delays, depDataPlc: depData,
            depDataDelays: delays})
        
        for i in range(numTimes): #load the cache
            iterOutput = sess.run(iterNext)
            for index, x in enumerate(datastore): x[i] = iterOutput[index]

        sess.run(iterUnb.initializer, feed_dict={arrDataUnb:arrData, arrDelaysUnb:delays, depDataUnb:depData,
                                depDataDelaysUnb:delays})
        arrDTrain, depDTrain, yDTrain = sess.run(iterUnbNext)

        sess.run(iterUnb.initializer, feed_dict={arrDataUnb:testArrData, arrDelaysUnb:testDelays, depDataUnb:testDepData,
                                depDataDelaysUnb:testDelays, useDropout:False})
        arrDTest, depDTest, yDTest = sess.run(iterUnbNext)

        # training loop
        for j in range(NUM_EPOCHS):
           a = time.time()
           for i in range(numSequences // BATCH_SIZE):
               arrD, depD, yD = [x[i] for x in datastore]
               sess.run(minimize, feed_dict={arrDataPlaceholder:arrD, depDataPlaceholder:depD, yPlaceholder:yD, useDropout:True})
           b = time.time()
           print("Epoch # {}".format(j), "finished in", b-a, "sec")
        
           if j % PRINT_RATE == 0:
                predictions, Y, l = sess.run((prediction, yData, loss), feed_dict={arrDataPlaceholder:arrDTrain,
                depDataPlaceholder:depDTrain, yPlaceholder:yDTrain, useDropout:False})
                
                calculateStatistics(np.concatenate((predictions, Y), axis=1))
                with open('lstmOutput/train-{}-{}.txt'.format(timestamp, j), 'w') as x:
                    calculateStatistics(np.concatenate((predictions, Y), axis=1), f=x) # l, f=x)

                predictions, Y, l = sess.run((prediction, yData, loss), feed_dict={arrDataPlaceholder:arrDTest,
                depDataPlaceholder:depDTest, yPlaceholder:yDTest, useDropout: False})
                
                calculateStatistics(np.concatenate((predictions, Y), axis=1)) # , l)
                with open('lstmOutput/test-{}-{}.txt'.format(timestamp, j), 'w') as x:
                    calculateStatistics(np.concatenate((predictions, Y), axis=1), f=x) # l, f=x)
                print(l)

if __name__ == "__main__":
    main()
