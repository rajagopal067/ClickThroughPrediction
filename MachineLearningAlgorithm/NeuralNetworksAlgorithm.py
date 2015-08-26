from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.metrics import confusion_matrix,roc_curve
import time
import sys
import pylab as pl
import numpy as np
import neurolab as nl
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,ClassificationDataSet
from sklearn.metrics import accuracy_score
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

import numpy as np
class NeuralNetworksAlgorithm:

    def __init__(self,train_file,test_file):
        self.train_file = train_file
        self.test_file = test_file

    def loadData(self,file_name):
        with open(file_name) as f:
            data = []
            for line in f:
                line = line.strip().split(",")
                data.append([x for x in line])

        return  data

    def neuralNetworksTrain(self):
        alldata = ClassificationDataSet( 23, 1, nb_classes=2)
        train_input_data = self.loadData(self.train_file)
        test_input_data = self.loadData(self.test_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        for i in range(0,len(features)):
            alldata.addSample(features[i], target[i])        
            
        tstdata, trndata = alldata.splitWithProportion(0.25)
        trndata._convertToOneOfMany()
        tstdata._convertToOneOfMany()
        
        INPUT_FEATURES = 23
        CLASSES = 2
        HIDDEN_NEURONS = 200
        WEIGHTDECAY = 0.1
        MOMENTUM = 0.1
        EPOCH = 2
        fnn = buildNetwork(trndata.indim, HIDDEN_NEURONS, trndata.outdim,outclass=LinearLayer)
        trainer = BackpropTrainer(fnn, dataset=trndata, momentum=MOMENTUM,verbose=True, weightdecay=WEIGHTDECAY)
        
        trainer.trainEpochs(EPOCH)
        pred = trainer.testOnClassData(dataset=tstdata)
        actual = tstdata['class']
        self.computeAccuracy(actual,pred)
        testFeatures = [x[2:] for x in test_input_data]
        testFeatures = testFeatures[1:]
        prediction = [fnn.activate(x) for x in testFeatures]
        i=0
        print "Neural Network Architecture:"
        print "Layers: Input layer, Hidden Layer and Output Layers"
        print "Epoch = "+str(EPOCH)
        print "Neurons in the hidden layer:"+str(HIDDEN_NEURONS)
        print "Precision recall F score support metrics for Neural Networks "
        print precision_recall_fscore_support(actual,pred)
        print "confusion matrix"
        print confusion_matrix(actual,pred)


    def computeAccuracy(self,predictedOutput,actualOutput):
        count = 0
        for i in range(len(predictedOutput)):
            if predictedOutput[i] == actualOutput[i]:
                count = count +1
        print "Accuracy for model is "
        print float(count)/float(len(predictedOutput))


print 'Neural Networks'
obj = NeuralNetworksAlgorithm('../clean_data/clean_train_1L.csv','../clean_data/clean_test_1k.csv')
obj.neuralNetworksTrain()
