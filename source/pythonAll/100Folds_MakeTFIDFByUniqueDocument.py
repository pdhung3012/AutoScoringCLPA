from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut

import pandas as pd
from scipy import spatial
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import nltk
from sklearn.metrics import accuracy_score
import os

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

def readStringFromFile(fpFile):
    f = open(fpFile, "r")
    content=f.read()
    f.close()
    return content
def printVector(vector):
    strOut=""
    i=0
    # print(len(vector))
    for item in vector:
        i=i+1
        strOut = strOut+ str(item)
        if(i!= len(vector)):
            strOut=strOut+ ","
    return strOut


def getTFIDFFromText(fpInputSubmission, fopTextFolder, fpOutputVectorTFIDF,fpPredict,fpExpect,fpSummarization):
    my_csv = pd.read_csv(fpInputSubmission)
    # filtered = my_csv.Score.str.match("I-",na=False)
    # my_csv3 = my_csv2[my_csv2.Score != "I-UR"]
    columnRevisedNetID = my_csv.RevisedNetID
    columnScore = my_csv.Score


    strA = 'A'
    strAMinus = 'A-'
    strBPlus = 'B+'
    strB = 'B'
    strBMinus = 'A-'


    print(str(len(columnRevisedNetID)))
    lenOfData=len(columnRevisedNetID)
    numOfTestPerFold=1
    print(str(numOfTestPerFold))

    csvTFIDF = open(fpOutputVectorTFIDF, 'w')
    lstResponses=[]

    dictScoreResponse = {strA: [], strAMinus: [], strBPlus: [], strB: [], strBMinus: []}
    for i in range(lenOfData):
        strScore = columnScore[i]
        item=columnRevisedNetID[i]
        fpTextItem = fopTextFolder + str(item) + "/text.txt"
        strResponse = readStringFromFile(fpTextItem)
        lstResponses.append(strResponse)
        if strScore == strA:
            dictScoreResponse[strA].append(strResponse)
        elif strScore == strAMinus:
            dictScoreResponse[strAMinus].append(strResponse)
        elif strScore == strBPlus:
            dictScoreResponse[strBPlus].append(strResponse)
        elif strScore == strB:
            dictScoreResponse[strB].append(strResponse)
        elif strScore == strBMinus:
            dictScoreResponse[strBMinus].append(strResponse)


    strTotalA = ' '.join(dictScoreResponse[strA])
    strTotalAMinus = ' '.join(dictScoreResponse[strAMinus])
    strTotalBPlus = ' '.join(dictScoreResponse[strBPlus])
    strTotalB = ' '.join(dictScoreResponse[strB])
    strTotalBMinus = ' '.join(dictScoreResponse[strBMinus])
    corpus = [str(strTotalA), str(strTotalAMinus), str(strTotalBPlus), str(strTotalB), str(strTotalBMinus)]
    for strResponse in lstResponses:
        corpus.append(strResponse)

    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(corpus)

    dictTopicVectors = {strA: [], strAMinus: [], strBPlus: [], strB: [], strBMinus: []}
    dictTopicVectors[strA] = X[0].todense()
    dictTopicVectors[strAMinus] = X[1].todense()
    dictTopicVectors[strBPlus] = X[2].todense()
    dictTopicVectors[strB] = X[3].todense()
    dictTopicVectors[strBMinus] = X[4].todense()

    columnTDFTitle ="no,reviseNetID,expected,"
    vector = X[5].toarray()[0]
    for i in range(len(vector)):
        columnTDFTitle = columnTDFTitle+ str((i+1))
        if i!=len(vector)-1:
            columnTDFTitle=columnTDFTitle+','
    columnTDFTitle = columnTDFTitle + '\n'
    csvTFIDF.write(columnTDFTitle)

        # print(str(len(corpus)))
    listPredictedResult=[]
    listExpectedResult = []
    for i in range(5, len(corpus)):
        # print(str(i)+"\tcontent here")
        indexCorpus=i-5
        expectedResult = columnScore[indexCorpus]
        listExpectedResult.append(expectedResult)
        strTestId = columnRevisedNetID[indexCorpus]
        vectori = X[i].todense()
        # ai = np.array(vectori)
        # print('damn {}'.format(len(ai[0])))
        # print('{}'.format(ai[0]))
        strVectorContent=printVector(X[i].toarray()[0])
        rowTFIDF=str(indexCorpus+1)+ ',' + str(strTestId)+',' + str(expectedResult) + ','+strVectorContent+ '\n'
        csvTFIDF.write(rowTFIDF)

        lstSim=[]
        lstScore=[]

        for j in range(5, len(corpus)):
            if j==i:
                # print('not count {}-{}'.format(i,j))
                continue
            vectorj=X[j].todense()
            simIJ=cosine_similarity(vectori, vectorj)[0][0]
            lstSim.append(simIJ)
            lstScore.append(columnScore[j-5])
        maxIndex=lstSim.index(max(lstSim))
        listPredictedResult.append(lstScore[maxIndex])
        print(lstSim[maxIndex])
    csvTFIDF.close()

    scoreTotalSim=accuracy_score(listExpectedResult, listPredictedResult)

    fp=open(fpPredict,'w')
    fp.write('\n'.join(listPredictedResult))
    fp.close()
    fp=open(fpExpect,'w')
    fp.write('\n'.join(listExpectedResult))
    fp.close()
    fp=open(fpSummarization,'w')
    fp.write('Total accuracy: {}\n'.format(scoreTotalSim))
    fp.close()




def main():
    fopInput="/home/hung/git/dataLocal/hw2VectorData/"
    fopText = "/home/hung/git/dataLocal/hw2TextualInformation/"
    fpStudentSubmission=fopInput+"StudentScore - ReviseList.csv"
    fpStudentDistance = fopInput + "vectorDistance_folds.csv"
    fpStudentTFIDF = fopInput + "vectorTFIDF_folds.csv"
    fopResult="/home/hung/git/dataLocal/hw2VectorData/resultCosineComparison/"
    fpExpected=fopResult+'expected.txt'
    fpPredicted = fopResult + 'predicted.txt'
    fpSum=fopResult+'summarization.txt'
    createDirectory(fopResult)

    getTFIDFFromText(fpStudentSubmission, fopText, fpStudentTFIDF,fpPredicted,fpExpected,fpSum)

main()
