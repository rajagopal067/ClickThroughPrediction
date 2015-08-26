import pandas as pd
from sklearn import tree
class PreProcessDataSet:
    def __init__(self,file_name):
        self.file_name = file_name
        tree.DecisionTreeClassifier

    def load_csv_file(self):
        self.dataFrame = pd.read_csv(self.file_name)

    def convert_obj_to_int(self):
        self.load_csv_file()
        df  = self.dataFrame
        object_list_columns = df.columns
        object_list_dtypes = df.dtypes
        new_col_suffix = '_int'
        for index in range(0,len(object_list_columns)):
            if object_list_dtypes[index] == object :
                df[object_list_columns[index]+new_col_suffix] = df[object_list_columns[index]].map( lambda  x: hash(x))
                df.drop([object_list_columns[index]],inplace=True,axis=1)

        df.to_csv('../clean_data/clean_train_10k.csv')


obj = PreProcessDataSet('/home/vijay/Matlab-Assignments/Project/ClickThroughPrediction/trainDataSets/train_10k.csv')
obj.convert_obj_to_int()
