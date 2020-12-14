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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


# set file directory
fpInput='/home/hung/git/dataLocal/hw2VectorData/'
fpOutput='/home/hung/git/dataLocal/hw2VectorData/analyzeCombine/'
createDirectory(fpOutput)
fnAll='vectorCombine_TFIDF_folds.csv'
# load data for 10-fold cv
df_all = pd.read_csv(fpInput+fnAll)
print(list(df_all.columns.values))
all_label = df_all['expected']
# all_data = df_all.drop(['label','maxSim','maxSim-r2','maxSim-r3','maxSim-r4','maxSim-p1','maxSim-p2','maxSim-p3','maxSim-p4'],axis=1)
all_data = df_all.drop(['no','reviseNetID','expected'],axis=1)

# create a list of classifiers
random_seed = 1234
classifiers = [GaussianNB(), LogisticRegression(random_state=random_seed),DecisionTreeClassifier(),
               RandomForestClassifier(random_state=random_seed, n_estimators=50), AdaBoostClassifier(), LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),
               LinearSVC(random_state=random_seed)]

# fit and evaluate for 10-cv
index = 0
# group = df_all['label']
o2=open(fpOutput+'result_features.txt','w')
k_fold = StratifiedKFold(10,shuffle=True)

arrAcc=[]
arrCName=[]

classifier=DecisionTreeClassifier()
index=0
filePredict=''.join([fpOutput,'predict_',str(index),'.txt'])
print("********", "\n", "10 fold CV Results with: ", str(classifier))
# cross_val = cross_val_score(classifier, all_data, all_label, cv=k_fold, n_jobs=1)
# predicted = cross_val_predict(classifier, all_data, all_label, cv=k_fold)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(all_data,all_label)
lstScores=fit.scores_
columns=all_data.columns
print(len(lstScores))
n=100
nMinus=-1*n
top_n_idx = sorted(range(len(lstScores)), key=lambda i: lstScores[i])[nMinus:]
# top_n_idx=reversed(top_n_idx)
# top_n_values = [lstScores[i] for i in top_n_idx]

print('{}\t{}\n'.format(lstScores[top_n_idx[0]],top_n_idx))

# lstFeatures=featureScores.nlargest(10,'Score')
# print(featureScores.nlargest(10,'Score'))  #print 10 best features
# totalAccScore=accuracy_score(all_label, predicted)
# cName=str(type(classifier)).split(".")[-1][:-2]
# arrCName.append(cName)
# arrAcc.append(totalAccScore)
# np.savetxt(filePredict,predicted,fmt='%s', delimiter=',')
o2.write('Result for top featuress of '+str(classifier)+'\n')
for i2 in range(0,len(top_n_idx)):
    i=len(top_n_idx)-1-i2
    o2.write('{}\t{}\t{}\t{}\n'.format((i+1),top_n_idx[i],columns[top_n_idx[i]],lstScores[top_n_idx[i]]))
# o2.write('Features: \n{}\n'.format('\n'.format(lstFeatures)))

o2.close()