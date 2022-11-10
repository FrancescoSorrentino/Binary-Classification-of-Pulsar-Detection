# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""
import numpy
import scipy.optimize
import DCF
import pylab
import features as f
import time

def mcol(v):
    #reshape a row vector in a column vector
    #!!!! if u write (v.size,) it will remain a ROW vector
    #So don't forget the column value "1"
    return v.reshape((v.size,1))

def mrow(v):
    return v.reshape((1, v.size))

def empirical_mean(D):
    return mcol(D.mean(1))

def empirical_cov(D, muc):
    DC = D - muc #class samples centered
    C = (numpy.dot(DC , DC.T))/D.shape[1]
    return C

def compute_PCA(D,m):
    #i calculate the means "mu" of all data.
    mu = empirical_mean(D)
    #mu is a row vector, so to center the dataset "D" i have
    #to make mu a column vector and then subtract it from
    #all column of D.
    DC = D - mu
    #now i can calculate the covariance matrix
    #making 1/N * (DC)*(DC).T
    C = (numpy.dot(DC , DC.T))/float(D.shape[1])
    #D.shape give us the number of value (n*m)
    s, U = numpy.linalg.eigh(C)
    #eigh compute the eigenvalues (ordered in increasing order ty to the "h" of the function)
    #and the corrisponding eigenvectors (columns of U).
    #but we will use svd to compute the Singualr Value Decompotition
    #and the eigenvalues sorted in decreasing order.
    U, s, Vh = numpy.linalg.svd(C)
    #we read the m hyperparameter m = int(input())
    #i make it 2 to have a 2 dimensional projection.
    #we use only m eigenvalues/eigenvectors (the first 5)
    P = U[:,0:m]
    #now we can calculate the projection matrix
    #DP = numpy.dot(P.T, D)
    
    return P
"""
D-> data training, m->numero di feat, X->matrice da trasformare
"""
def PCA(P,X):
    DP=numpy.dot(P.T,X)
    return DP

def train_SVM_RBF(DTR, LTR, C, gamma, p = 0, K = 1):
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2 * numpy.dot(DTR.T, DTR)
    H = numpy.exp(-gamma*Dist)
    H = mcol(Z) * mrow(Z) * H
    alphaStar = compute_opt(H,DTR,LTR,C,p)
 
    return mcol(alphaStar) * mcol(Z)

def compute_opt(H, DTR, LTR, C, p):
    def JDual(alpha):
        Ha= numpy.dot(H,mcol(alpha))
        aHa= numpy.dot(mrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    empP = (LTR == 1).sum()/len(LTR) 
    bounds = numpy.array([(0, C)] * LTR.shape[0])
    if p != 0: 
        bounds[LTR == 1] = (0, C*p/empP) 
        bounds[LTR == 0] = (0, C*(1-p)/(1-empP))
        
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b( LDual, numpy.zeros(DTR.shape[1]), bounds = bounds, iprint = 1,factr = 1.0 ,maxiter = 100000,
        maxfun=100000,)
    
    return alphaStar
 
def compute_scores(DTR, DTE, gamma, wStar):
    Dist = mcol((DTR**2).sum(0)) + mrow((DTE**2).sum(0)) - 2 * numpy.dot(DTR.T, DTE)
    k = numpy.exp(-gamma*Dist)
    
    return numpy.dot(wStar.T, k)

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

def rbf_svm(D,L,DE,LE, C, gamma, p = 0):
    wStar = train_SVM_RBF(D, L, C, gamma, p)
    scores = compute_scores(D,DE,gamma,wStar).sum(0).tolist()
    
    return scores
    
def BestRBF(D,L,k, gaussianize=0,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1])
    scores = []
    labels = []
    scoresPerPrior = []
    labelsPerPrior = []
    priors = [0.5,0.1,0.9]
    for i in range(k):
        #print("Fold :", i+1)
        idxTrain = idx[0:nTrain] 
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain] 
        DTE = D[:, idxTest]
        if gaussianize==1:
            DTR = f.gaussianize_features(DTR, DTR)
            DTE = f.gaussianize_features(DTR, DTE)
        LTR = L[idxTrain] 
        LTE = L[idxTest]
        #Applico PCA
        P=compute_PCA(DTR,7)
        DTR=PCA(P,DTR)
        DTE=PCA(P,DTE)
        labels.append(LTE.tolist())
        wStar = train_SVM_RBF(DTR, LTR, 10, 1e-2,0.5)
        scores.extend(compute_scores(DTR,DTE,1e-2,wStar).sum(0).tolist())
        #scores.extend(trainLinearSVM(DTR, LTR, 1, 1e-3, DTE))
        idx = numpy.roll(idx,nTest,axis=0)
        #if gaussianized == 1:
            #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        #else :
    for pi in priors:
        print('minDCF SVM with C ', 10 ,', gamma ', 0.01 , ', and application ', pi,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
    #print('minDCF SVM with C ', 10 ,', gamma ', 0.01 , ', and application ', pi,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        #scoresPerPrior.append(scores)
        #labelsPerPrior.append(numpy.hstack(labels))
        #if pi == 0.5 and pi_T == 0.5:
            #DCF.plot_minDCF(scores, numpy.hstack(labels))
        

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return scores, numpy.hstack(labels)

def k_fold(D,L,k, gaussianize=0,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1])
    scores = []
    labels = []
    priors = [0.5,0.1,0.9]
    for i in range(k):
        #print("Fold :", i+1)
        idxTrain = idx[0:nTrain] 
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain] 
        DTE = D[:, idxTest]
        if gaussianize==1:
            DTR = f.gaussianize_features(DTR, DTR)
            DTE = f.gaussianize_features(DTR, DTE)
        LTR = L[idxTrain] 
        LTE = L[idxTest]
        #Applico PCA
        #P=compute_PCA(DTR,7)
        #DTR=PCA(P,DTR)
        #DTE=PCA(P,DTE)
        labels.append(LTE.tolist())
        wStar = train_SVM_RBF(DTR, LTR, 10, 1e-2)
        scores.extend(compute_scores(DTR,DTE,1e-2,wStar).sum(0).tolist())
        #scores.extend(trainLinearSVM(DTR, LTR, 1, 1e-3, DTE))
        idx = numpy.roll(idx,nTest,axis=0)
        #if gaussianized == 1:
            #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        #else :
    for pi in priors:
        print('minDCF SVM with C ', 10 ,', gamma ', 0.01 , ', and application ', pi,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        #if pi == 0.5 and pi_T == 0.5:
            #DCF.plot_minDCF(scores, numpy.hstack(labels))
        

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return scores

def Ctuning(D,L,k, gaussianize=0,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx_ = numpy.random.permutation(D.shape[1])
    # _ , DP = compute_PCA(D,7)
    C = numpy.logspace(-4,1,num=20, base=10.0)
    print(C)
    scores = []
    labels = []
    y = []
    gamma = [1e-2,1e-1,1]
    for g in gamma:
        scores = []
        labels = []
        minDCFs = []
        idx = idx_
        for c in C:
            scores = []
            labels = []
            idx = idx_
            for i in range(k):
                #print("Fold :", i+1)
                idxTrain = idx[0:nTrain] 
                idxTest = idx[nTrain:]
                DTR = D[:, idxTrain] 
                DTE = D[:, idxTest]
                if gaussianize==1:
                    DTR = f.gaussianize_features(DTR, DTR)
                    DTE = f.gaussianize_features(DTR, DTE)
                LTR = L[idxTrain] 
                LTE = L[idxTest]
                labels.append(LTE.tolist())
                wStar = train_SVM_RBF(DTR, LTR, c, g)
                scores.extend(compute_scores(DTR,DTE,g,wStar).sum(0).tolist())
                idx = numpy.roll(idx,nTest,axis=0)
            print('C: ', c)
            minDCFs.append(DCF.compute_min_DCF(scores, numpy.hstack(labels), 0.5, 1, 1))
        y.append(numpy.hstack(minDCFs))
        print(y)
    DCF.plot_DCF_gamma(C, y, 'C', 'gamma-C-tuning_RBF_PCA7_prekfold.svg')
            #if gaussianized == 1:
                #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
            #else :
            #print('minDCF SVM with C ', 1.0 ,' and application ', pi,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
            #if pi == 0.5 and pi_T == 0.5:
                #DCF.plot_minDCF(scores, numpy.hstack(labels))
        

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return scores

if __name__ == '__main__':
    #svm
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    ZD = f.ZNormalization(D)
    k_fold(ZD,L,3)