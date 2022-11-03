from os import name
from numpy import inexact, zeros
from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import tile

def file2matrix(filename):     #文件打开函数
    fr = open(filename)
    numberoflines = len(fr.readlines())
    returnmat = zeros((numberoflines,3))
    classlabelvector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listformline = line.split('\t')
        returnmat[index,:]= listformline[0:3]
        classlabelvector.append((listformline[-1]))
        index += 1
    return returnmat,classlabelvector
def classify0(inx,dataSet,labels,k):  #求距离函数
    import numpy as np
    import operator
    datasetsize = dataSet.shape[0]
    diffmat = tile(inx,(datasetsize,1)) - dataSet
    sqdiffmat = diffmat**2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances**0.5
    sorteddistindicies = distances.argsort()
    classcount = {}
    for i in range(k):
        voteilabel = labels[sorteddistindicies[i]]
        classcount[voteilabel] = classcount.get(voteilabel,0)+1
    sortedclasscount = sorted(classcount.items(),key =operator.itemgetter(1),reverse= True)
    return sortedclasscount[0][0]
def autoNorm(dataSet):       #标准化函数
    import numpy as np
    import numpy as np
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
def datingclasstest():
    horatio = 0.20
    datingdatamat, datinglabels = file2matrix("C:\\Users\\pc\\Desktop\\py\\KNN\\datingTestSet.txt.txt") 
    normDataSet, ranges, minVals = autoNorm(datingdatamat)  
    m = normDataSet.shape[0]   
    numtestvecs = int(m*horatio)  
    errorcount = 0.0
    for i in range(numtestvecs):
        classifierresult = classify0(normDataSet[i, : ], normDataSet[numtestvecs:m, :],  
                                    datinglabels[numtestvecs:m], 4)
        print("the classifier came back with: %s, the real answer is : %s" %
              (classifierresult, datinglabels[i]))
        if (classifierresult != datinglabels[i]):
            errorcount += 1.0
    print("the total right rate is : %d" % (1-errorcount/float(numtestvecs)))
datingclasstest()




      