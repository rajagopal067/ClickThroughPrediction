from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,roc_curve
import StringIO
from sklearn import svm
import time

import numpy as np
class RandomForesTreeAlgorithm:
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

    def getFeatureset(self,file_name):
        #load only feature sets from Data set ( excluding label )
        input_data = self.loadData(file_name)
        features = [x[2:] for x in input_data]
        features = features[1:]
        return features



    def getTargetset(self,file_name):
        #load only target labels from Data set
        input_data = self.loadData(file_name)
        target = [x[1] for x in input_data]
        target = target[1:]#Second column is target label
        return target


    def learnRandomForest(self):

        train_feature = self.getFeatureset(self.train_file)
        train_target = self.getTargetset(self.train_file)
        # classify the model using training set
        model = self.classify_random_forest(train_feature,train_target)

        test_feature = self.getFeatureset(self.test_file)
        test_target = self.getTargetset(self.test_file)
        predictedOutput = model.predict(test_feature)
        X= []
        Y=[]
        for a in predictedOutput:
            X.append(int(a))
        for a in test_target:
            Y.append(int(a))
        result = zip(Y,X)
        self.write_To_File(result,"cart-predictions.csv")
        self.computeAccuracy(predictedOutput,test_target)
        print "Precision recall F score support metrics for CART "
        print precision_recall_fscore_support(test_target,predictedOutput)
        print "confusion matrix"
        print confusion_matrix(test_target,predictedOutput)


    def write_To_File(self,result,filename):
        f = open(filename,'w')
        for pair in result:
            f.write(str(pair[0]) + "\t" + str(pair[1]) )
            f.write("\n")
        f.close()

    def classify_random_forest(self,X,Y):
       rf = RandomForestClassifier(n_estimators=400,max_features='log2')
       return rf.fit(X,Y)



    def printDTRules(self,model):
        dot_data = StringIO.StringIO()
        #with open("rules_1L.dot","w") as output_file:
        out = tree.export_graphviz(model, out_file="rules_1L.dot")




    def computeAccuracy(self,predictedOutput,actualOutput):
        count = 0
        for i in range(len(predictedOutput)):
            if predictedOutput[i] == actualOutput[i]:
                count = count +1
        print "Accuracy for model is "
        print float(count)/float(len(predictedOutput))

    def extractunique(self,data_set):
        columns = list()
        for t in range(0,len(data_set[0])):
            columns.append([])

        for x in data_set:
            for index,y in enumerate(x):
                columns[index].append(y)
        for index,x in enumerate(columns):
            columns[index] = set(columns[index])
        return columns


print "\nRandom Forests\n"
obj = RandomForesTreeAlgorithm('../clean_data/clean_train_1L.csv','../clean_data/clean_test_1k.csv')
obj.learnRandomForest()

start_time = time.time()

obj.learnRandomForest()
time_elapsed = time.time() - start_time
print "Time taken " + str(time_elapsed)
train_feature = obj.getFeatureset('../trainDataSets/train_1L.csv')
f1=open('testfile', 'w+')

for res in obj.extractunique(train_feature):
    print >> f1 , res