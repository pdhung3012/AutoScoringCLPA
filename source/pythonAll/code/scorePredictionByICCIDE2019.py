# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 18:19:24 2019

@author: hungphd
"""


# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
import os

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


# set file directory
fpInput='/home/hung/git/dataLocal/hw2VectorData/'
fpOutput='/home/hung/git/dataLocal/hw2VectorData/resultCosine/'
createDirectory(fpOutput)
fnAll='vectorTFIDF_folds.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fnAll)
print(list(df_all.columns.values))
all_label = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','reviseNetID','expected'],axis=1)

# fit and evaluate for 10-cv
index = 0
# group = df_all['label']
o2=open(fpOutput+'result_all.txt','w')

npVectors=all_data.to_numpy()
print('{}'.format(npVectors))
# for index in range(0,len(all_data)):


o2.close()