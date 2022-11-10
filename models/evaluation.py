# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:37:54 2022

@author: fra_s
"""

import MVG as mvg
import LR as lr
import QuadLR as qlr
import SVM as svm
import PolySVM as psvm
import RBFSVM as rbf
import GMM as gmm
import DCF as dcf
import features as f
import Fusion as fusion
import calibration as c
import numpy


def mcol(v):
    #reshape a row vector in a column vector
    #!!!! if u write (v.size,) it will remain a ROW vector
    #So don't forget the column value "1"
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1, v.size))

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

def minDCF(scores, LE):
    pi=[0.5,0.1,0.9]
    for p in pi:
        print('minDCF with application ', p,' :' , "%.3f" % dcf.compute_min_DCF(scores, LE, p, 1, 1))
    

if __name__ == '__main__':
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    print(DE.shape)
    mean = D.mean(axis=1)
    standardDeviation = D.std(axis=1)
    #Z-Score and Gaussianized 
    ZDT = f.ZNormalization(D)
    GDT = f.gaussianize_features(ZDT, ZDT)
    ZDE = f.ZNormalization(DE, mean, standardDeviation)
    GDE = f.gaussianize_features(ZDT, ZDE)
    #=======================
    #PCA m=7 Z-score
    P7 = f.compute_PCA(ZDT, 7)
    ZDT7 = f.PCA(P7, ZDT)
    ZDE7 = f.PCA(P7, ZDE)
    #======================
    #PCA m=6 Z-score
    P6 = f.compute_PCA(ZDT, 6)
    ZDT6 = f.PCA(P7, ZDT)
    ZDE6 = f.PCA(P7, ZDE)
    #======================
# =============================================================================
    #Evaluation on MVG Classifiers
    print('MVG Z-Score Normalization')
    print('MVG Full Cov')
    scores = mvg.log_mvg(ZDT, L, ZDE, LE)
    minDCF(scores, LE)
    print('MVG Tied Cov')
    scores = mvg.log_TCG(ZDT, L, ZDE, LE)
    minDCF(scores, LE)
    print('MVG Diag Cov')
    scores = mvg.log_naive_bayes(ZDT, L, ZDE, LE)
    minDCF(scores, LE)
    print('MVG Z-Score Normalization + PCA m=7')
    print('MVG Full Cov')
    scores = mvg.log_mvg(ZDT7, L, ZDE7, LE)
    minDCF(scores, LE)
    print('MVG Tied Cov')
    scores = mvg.log_TCG(ZDT7, L, ZDE7, LE)
    minDCF(scores, LE)
    print('MVG Diag Cov')
    scores = mvg.log_naive_bayes(ZDT7, L, ZDE7, LE)
    minDCF(scores, LE)
    print('MVG Z-Score Normalization + PCA m=6')
    print('MVG Full Cov')
    scores = mvg.log_mvg(ZDT6, L, ZDE6, LE)
    minDCF(scores, LE)
    print('MVG Tied Cov')
    scores = mvg.log_TCG(ZDT6, L, ZDE6, LE)
    minDCF(scores, LE)
    print('MVG Diag Cov')
    scores = mvg.log_naive_bayes(ZDT6, L, ZDE6, LE)
    minDCF(scores, LE)
    print('MVG Gaussianized features')
    print('MVG Full Cov')
    scores = mvg.log_mvg(GDT, L, GDE, LE)
    minDCF(scores, LE)
    print('MVG Tied Cov')
    scores = mvg.log_TCG(GDT, L, GDE, LE)
    minDCF(scores, LE)
    print('MVG Diag Cov')
    scores = mvg.log_naive_bayes(GDT, L, GDE, LE)
    minDCF(scores, LE)
#     #======================
    #Evaluation on LR Classifiers
    print('LR Z-Score Normalization')
    print('LR lamda=1e-5, pi_T = 0.5')
    scores = lr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.1')
    scores = lr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.9')
    scores = lr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.9)
    minDCF(scores, LE)
    print('LR Z-Score Normalization + PCA m=7')
    print('LR lamda=1e-5, pi_T = 0.5')
    scores = lr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.1')
    scores = lr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.9')
    scores = lr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.9)
    minDCF(scores, LE)
    print('LR Gaussianized features')
    print('LR lamda=1e-5, pi_T = 0.5')
    scores = lr.logreg(GDT, L, GDE, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.1')
    scores = lr.logreg(GDT, L, GDE, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('LR lamda=1e-5, pi_T = 0.9')
    scores = lr.logreg(GDT, L, GDE, LE, 1e-5, 0.9)
    minDCF(scores, LE)
#     #======================
    #Evaluation on Quad LR Classifiers
    print('Quad LR Z-Score Normalization')
    print('Quad LR lamda=1e-5, pi_T = 0.5')
    scores = qlr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.1')
    scores = qlr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.9')
    scores = qlr.logreg(ZDT, L, ZDE, LE, 1e-5, 0.9)
    minDCF(scores, LE)
    print('QuadLR Z-Score Normalization + PCA m=7')
    print('QuadLR lamda=1e-5, pi_T = 0.5')
    scores = qlr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.1')
    scores = qlr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.9')
    scores = qlr.logreg(ZDT7, L, ZDE7, LE, 1e-5, 0.9)
    minDCF(scores, LE)
    print('QuadLR Gaussianized features')
    print('QuadLR lamda=1e-5, pi_T = 0.5')
    scores = qlr.logreg(GDT, L, GDE, LE, 1e-5, 0.5)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.1')
    scores = qlr.logreg(GDT, L, GDE, LE, 1e-5, 0.1)
    minDCF(scores, LE)
    print('QuadLR lamda=1e-5, pi_T = 0.9')
    scores = qlr.logreg(GDT, L, GDE, LE, 1e-5, 0.9)
    minDCF(scores, LE)
#     #======================
# =============================================================================
# =============================================================================
    #Evaluation on SVM
    print('Linear SVM Z-Score Normalization')
    print('SVM C=0.1')
    scores = svm.linear_svm(ZDT, L, ZDE, LE, 1e-1)
    minDCF(scores,LE)
    print('Linear SVM Z-Score Normalization + PCA m=7')
    print('SVM C=0.1')
    scores = svm.linear_svm(ZDT7, L, ZDE7, LE, 1e-1)
    minDCF(scores,LE)
    print('Linear SVM Gaussianize features')
    print('SVM C=0.1')
    scores = svm.linear_svm(GDT, L, GDE, LE, 1e-1)
    minDCF(scores,LE)
    #======================
    #Evaluation on Quad SVM
    print('Quad SVM Z-Score Normalization')
    print('QSVM C=0.1')
    scores = psvm.quad_svm(ZDT, L, ZDE, LE, 1e-1)
    minDCF(scores,LE)
    print('Quad SVM Z-Score Normalization + PCA m=7')
    print('QSVM C=0.1')
    scores = psvm.quad_svm(ZDT7, L, ZDE7, LE, 1e-1)
    minDCF(scores,LE)
    print('Quad SVM Gaussianize features')
    print('QSVM C=0.1')
    scores = psvm.quad_svm(GDT, L, GDE, LE, 1e-1)
    minDCF(scores,LE)
    #======================
#=============================================================================
#=============================================================================
    #Evaluation on RBF SVM
    print('RBF SVM Z-Score Normalization')
    print('RBF C=10, gamma=0.01')
    scores = rbf.rbf_svm(ZDT, L, ZDE, LE, 10, 0.01)
    minDCF(scores,LE)
    print('RBF SVM Z-Score Normalization + PCA m=7')
    print('RBF C=10, gamma=0.01')
    scores = rbf.rbf_svm(ZDT7, L, ZDE7, LE, 10, 0.01)
    minDCF(scores,LE)
    print('RBF SVM Z-Score Normalization + PCA m=7 BALANCED')
    print('RBF C=10, gamma=0.01, pi_T = 0.5')
    scores = rbf.rbf_svm(ZDT7, L, ZDE7, LE, 10, 0.01, 0.5)
    minDCF(scores,LE)
    print('RBF SVM Gaussianize features')
    print('RBF C=10, gamma=0.1')
    scores = rbf.rbf_svm(GDT, L, GDE, LE, 10, 0.1)
    minDCF(scores,LE)
    #======================
    #Evaluation on GMM
    print('GMM Z-Score Normalization')
    print('GMM Full-Cov 8c')
    scores = gmm.GMM_Full(ZDT,ZDE,L,0.1,2**3,'Full').tolist()
    minDCF(scores, LE)
    print('GMM Tied 32c')
    scores = gmm.GMM_Full(ZDT,ZDE,L,0.1,2**5,'Tied').tolist()
    minDCF(scores, LE)
    print('GMM Diag 32c')
    scores = gmm.GMM_Full(ZDT,ZDE,L,0.1,2**5,'Diag').tolist()
    minDCF(scores, LE)
    print('GMM Tied-Diag 64')
    scores = gmm.GMM_Full(ZDT,ZDE,L,0.1,2**6,'TiedDiag').tolist()
    minDCF(scores, LE)
    #=========================
# =============================================================================
    #Evaluation of calibrated best models + fusion
    scores2 = mvg.log_TCG(ZDT7, L, ZDE7, LE).tolist()
    print(len(scores2))
    print(LE.shape)
    scores1 = rbf.rbf_svm(ZDT7, L, ZDE7, LE, 10, 0.01, 0.5)
    c.computeScores(ZDT7, L, scores1, scores2, LE)
    
    
    