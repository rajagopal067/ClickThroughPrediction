import StringIO
import time

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import cross_validation
import numpy


class SVMClassifierAlgorithm:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file


    def classify(self,X,Y,final_c):
        svmmodel = svm.SVC(C=final_c)
        return svmmodel.fit(X,Y)

    def loadData(self, file_name):
        with open(file_name) as f:
            data = []
            for line in f:
                line = line.strip().split(",")
                data.append([x for x in line])

        return data

    def getFeatureset(self, file_name):

        input_data = self.loadData(file_name)
        features = [x[2:] for x in input_data]
        features = features[1:]
        return features

    def getTargetset(self, file_name):

        input_data = self.loadData(file_name)
        target = [x[1] for x in input_data]
        target = target[1:]
        return target


    def learn_svm(self):
        train_feature_1 = self.getFeatureset('../clean_data/clean_train_10k.csv')
        train_target = self.getTargetset('../clean_data/clean_train_10k.csv')
        # classify the model using training set

        #let's do cross validation !!!!

        k_fold = cross_validation.KFold(n=len(train_feature_1), n_folds=10)
        #confusion_matrix=[]
        res = numpy.zeros((2,2))
        accuracy = dict()
        cvalue = [0.00001,0.0001,0.001,0.01,0.1,1,10,1000,10000,100000 ]
        for c in cvalue:
         acc = 0.0
         for train_indices, test_indices in k_fold:
            feature = self.transform_data(train_feature_1,train_indices)
            target = self.transform_data(train_target,train_indices)
            model = self.classify(feature,target,c)
            test_feature =  self.transform_data(train_feature_1,test_indices)
            test_target = self.transform_data(train_target,test_indices)
            predictedOutput = model.predict(test_feature)
            acc = acc + self.computeAccuracy(predictedOutput,test_target)
         accuracy[c] = acc/10.0
         print "Accuracy for model is ",accuracy[c]
        max = accuracy[0.00001]
        final_c = 0.00001
        for c,value in accuracy.items():
            if value > max:
                max = value
                final_c = c
        #use the final C value:
        print final_c
        train_input_data = self.loadData(self.train_file)
        target = [x[1] for x in train_input_data]
        target = target[1:]
        features = [x[2:] for x in train_input_data]
        features = features[1:]
        model = self.classify(features,target,final_c)

        test_input_data = self.loadData(self.test_file)
        actualOutput = [x[1] for x in test_input_data]
        actualOutput = actualOutput[1:]
        features = [x[2:] for x in test_input_data]
        features = features[1:]

        predictedOutput = model.predict(features)
        #print predictedOutput
        #print actualOutput
        X= []
        Y=[]
        for a in predictedOutput:
            X.append(int(a))
        for a in actualOutput:
            Y.append(int(a))
        result = zip(Y,X)
        self.write_To_File(result,"cart-predictions.csv")
        self.computeAccuracy(predictedOutput,actualOutput)
        print "Precision recall Fscore support metrics for CART "
        print precision_recall_fscore_support(actualOutput,predictedOutput)
        print "\nconfusion matrix\n"
        print confusion_matrix(actualOutput,predictedOutput)

    def write_To_File(self,result,filename):
        f = open(filename,'w')
        for pair in result:
            f.write(str(pair[0]) + "\t" + str(pair[1]) )
            f.write("\n")
        f.close()

    def transform_data(self,dataset,index_array):
        transformedDataset = []
        for val in index_array:
            transformedDataset.append(dataset[val])
        return transformedDataset
    def computeAccuracy(self, predictedOutput, actualOutput):
        count = 0
        for i in range(len(predictedOutput)):
            if predictedOutput[i] == actualOutput[i]:
                count = count + 1
        ac = float(count) / float(len(predictedOutput))
        return ac

print "\n SVM \n"
obj = SVMClassifierAlgorithm('../clean_data/clean_train_1L.csv', '../clean_data/clean_test_10k.csv')
obj.learn_svm()