# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""

import numpy 
import scipy.special 
import DCF
import features as f
import RBFSVM as svm
import MVG as mvg
import LR as lr
import calibration as c
def mcol(v):
    #reshape a row vector in a column vector
    #!!!! if u write (v.size,) it will remain a ROW vector
    #So don't forget the column value "1"
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1, v.size))

#==============================================================================
# --------- LOAD FILE ---------------------------------------------------------
def load(fname):
    #we have 9 parameters per row
    #8 values and one label
    #we take the 8 values and put em in a column format into DList
    #we take the last argument and put it in the labelsList
    #so we have a 1:1 association index between the column and the label
    
    DList = []
    labelsList = []
    with open(fname) as f:
        for line in f:
            try:
               #i take the first 4 numbers as a vector
               attrs = line.split(",")[0:8]
               #i parse the number from string to float, then i
               #transform it in a column vector
               attrs = mcol(numpy.array([float(i) for i in attrs])) 
               label = int(line.split(",")[-1].strip())
               DList.append(attrs)
               labelsList.append(label)
            except:
                pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype = numpy.int32)
#==============================================================================
def logRegModel(D,L,  seed=0):
    nTrain = int(D.shape[1]*2/3)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1])
    # _ , DP = compute_PCA(D,7)
    scores = []
    labels = []
    priors = [0.5]
# =============================================================================
#     idx = idx
#     idxTrain = idx[0:nTrain] 
#     idxTest = idx[nTrain:]
#     DTR = D[:, idxTrain] 
#     DTE = D[:, idxTest]
#     LTR = L[idxTrain] 
#     LTE = L[idxTest]
#     labels.append(LTE.tolist())
# =============================================================================
    #scores.extend(lr.logreg(DTR, LTR, DTE, LTE, 0, 0.5))
    for i in range(3):
        #print("Fold :", i+1)
        idxTrain = idx[0:nTrain] 
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain] 
        DTE = D[:, idxTest]
        LTR = L[idxTrain] 
        LTE = L[idxTest]
        #Applico PCA
        #P=compute_PCA(DTR,7)
        #DTR=PCA(P,DTR)
        #DTE=PCA(P,DTE)
        ##############
        labels.append(LTE.tolist())
        scores.extend(lr.logreg(DTR, LTR, DTE, LTE, 0, 0.5))
        idx = numpy.roll(idx,nTest,axis=0)
        #scores = lr.logreg(D, L, D, L, 0, 0.5)
        
    print('minDCF LR with prior ', 0.5 ,' and application ', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), 0.5 , 1, 1))
        
    return scores, numpy.hstack(labels)

def logRegModel2(DTR,LTR,DTE,LTE, seed=0):
    
    labels = LTE.tolist()
    _w, _b1 = lr.logreg2(DTR, LTR, DTE, LTE, 0, 0.5)
    scores = numpy.dot(_w.T, DTE) + _b1
        
    #print('minDCF LR with prior ', 0.5 ,' and application ', 0.5,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, LTE, 0.5 , 1, 1))
        
    return scores
   
def fusion_function(scores1,scores2,DE, labels, LE):
    
    
    DTR = numpy.stack((scores2, scores1))
    scoresfused= logRegModel2(DTR, labels,DE,LE)
    
    
    
    return scoresfused

def fusion_function2(scores1,scores2, labels):
    
    
    DTR = numpy.stack((scores2, scores1))
    return logRegModel(DTR, labels)
    
    
    
    #return scoresfused,labels


def fusionModel(scores1, scores2, DE, labels, LE):
    
    #scores1, labels = svm.BestRBF(D, L, k)
    #scores2, labels2 = mvg.BestMVG(D, L, k)
    #scores1 = c.calibrateScore(mrow(numpy.array(scores1)), labels)
    #scores2 = c.calibrateScore(mrow(numpy.array(scores2)), labels)
    print(LE.shape)
    scores= fusion_function(scores1, scores2, DE, labels,LE)
    #scores = fusion_f2(scores1, scores2)
    #DCF.plot_minDCF(scores, labels, 'prova_fusion.svg')
    pi = [0.5,0.1,0.9]
    print('min')
    for p in pi:
        print(DCF.compute_min_DCF(scores, LE, p, 1, 1))
        #print(DCF.compute_min_DCF(scores, labels, p, 1, 1))
    print('Act')
    for p in pi:
        print(DCF.compute_act_DCF(scores, LE, p, 1, 1))
        #print(DCF.compute_act_DCF(scores, labels, p, 1, 1))
    return scores

def fusionModel2(scores1, scores2, labels):
    
    #scores1, labels = svm.BestRBF(D, L, k)
    #scores2, labels2 = mvg.BestMVG(D, L, k)
    #scores1 = c.calibrateScore(mrow(numpy.array(scores1)), labels)
    #scores2 = c.calibrateScore(mrow(numpy.array(scores2)), labels)
    scores, lab= fusion_function2(scores1, scores2,labels)
    #scores = fusion_f2(scores1, scores2)
    #DCF.plot_minDCF(scores, labels, 'prova_fusion.svg')
    pi = [0.5,0.1,0.9]
    print('min')
    for p in pi:
        print(DCF.compute_min_DCF(scores, lab, p, 1, 1))
        #print(DCF.compute_min_DCF(scores, labels, p, 1, 1))
    print('Act')
    for p in pi:
        print(DCF.compute_act_DCF(scores, lab, p, 1, 1))
        #print(DCF.compute_act_DCF(scores, labels, p, 1, 1))
    return scores, lab
     
if __name__ == '__main__':
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    #_ = k_fold(D,L,5)
    ZD = f.ZNormalization(D)
    fusionModel(ZD, L, 3)