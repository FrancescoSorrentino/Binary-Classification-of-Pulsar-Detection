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
import Fusion as fusion

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

def calibrateScore(D,L, seed=0):
    # _ , DP = compute_PCA(D,7)
    scores = lr.logreg_cal(D, L, D, L, 1e-5, 0.5)
    
    pi = [0.5,0.1,0.9]
    for p in pi:
        print(DCF.compute_act_DCF(scores, L, p, 1, 1))
    
    return scores

def calibrateScore2(D,L,scores,LE, seed=0):
    # _ , DP = compute_PCA(D,7)
    scores = lr.logreg_cal(D, L, scores, LE, 1e-5, 0.5)
    
    pi = [0.5,0.1,0.9]
    for p in pi:
        print(DCF.compute_act_DCF(scores, LE, p, 1, 1))
    
    return scores

def computeScores(D,L,scores1,scores2,LE):
    svmsc , labels = svm.BestRBF(D, L, 3)
    mvgsc, labels2 = mvg.BestMVG(D, L, 3)
    #DCF.plot_minDCF(svmsc, labels, 'SvmRbf_dcf.svg')
    #DCF.plot_minDCF(mvgsc, labels, 'Mvg_scf.svg')
    print('svm')
    svmsc1 = calibrateScore(mrow(numpy.array(svmsc)), labels)
    print('mvg')
    mvgsc1 = calibrateScore(mrow(numpy.array(mvgsc)), labels)
    print('cal eval, mvg and svm')
    scores2 = calMVG(mvgsc, labels2, scores2, LE)
    scores1 = calSVM(svmsc, labels2, scores1, LE)
    DE = numpy.stack((scores2, scores1))
    fuscal = fusion.fusionModel(svmsc1, mvgsc1, DE, labels, LE)
    DCF.plot_DET(scores1, scores2, fuscal, LE,LE, 'DET-ev.svg')
    DCF.plot_ROC(scores1, scores2, fuscal, LE,LE, 'ROC-ev.svg')
    #DCF.plot_minDCF_best(mvgsc, mvgsc, svmsc, svmsc, labels, 'svm+mvg_noCAL.svg')
    #DCF.plot_minDCF_final(mvgsc1,svmsc1, fuscal, labels, labels3, 'fusioncomp_Finaliguess.svg')

def calSVM(D,L,scores,LE):
    #svmsc , labels = svm.BestRBF(D, L, 3)
    return calibrateScore2(mrow(numpy.array(D)),L,mrow(numpy.array(scores)),LE)
    
def calMVG(D,L,scores,LE):
    #mvgsc, labels2 = mvg.BestMVG(D, L, 3)
    return calibrateScore2(mrow(numpy.array(D)),L,mrow(numpy.array(scores)),LE)
    
def computeScores2(D,L):
    svmsc , labels = svm.BestRBF(D, L, 3)
    mvgsc, labels2 = mvg.BestMVG(D, L, 3)
    #DCF.plot_minDCF(svmsc, labels, 'SvmRbf_dcf.svg')
    #DCF.plot_minDCF(mvgsc, labels, 'Mvg_scf.svg')
    print('svm')
    svmsc1 = calibrateScore(mrow(numpy.array(svmsc)), labels)
    print('mvg')
    mvgsc1 = calibrateScore(mrow(numpy.array(mvgsc)), labels)
    print('cal eval, mvg and svm')
    fuscal,lab = fusion.fusionModel2(svmsc1, mvgsc1,labels)
    DCF.plot_DET(svmsc1, mvgsc1, fuscal, labels,lab, 'DET.svg')
    DCF.plot_ROC(svmsc1, mvgsc1, fuscal, labels,lab, 'ROC.svg')
    #DCF.plot_minDCF_best(mvgsc, mvgsc, svmsc, svmsc, labels, 'svm+mvg_noCAL.svg')
    #DCF.plot_minDCF_final(mvgsc1,svmsc1, fuscal, labels, labels3, 'fusioncomp_Finaliguess.svg')    

if __name__ == '__main__':
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    #_ = k_fold(D,L,5)
    ZD = f.ZNormalization(D)
    computeScores2(ZD, L)