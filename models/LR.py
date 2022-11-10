# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""
import numpy
import scipy.optimize
import DCF
import pylab
import features as f

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

def logreg_obj_wrap(DTR, LTR, l, prior): 
    Z0 = -1
    Z1 = 1
    M = DTR.shape[0]
    nt = DTR[:,LTR==1].shape[0]
    nf = DTR[:,LTR==0].shape[0]
    def logreg_obj(v): 
        # ... 
        # Compute and return the objective function value using DTR, LTR, l
        # ... 
        w = mcol(v[0:M])
        b = v[-1]
        #scores
        S0 = numpy.dot(w.T, DTR[:,LTR==0]) + b
        S1 = numpy.dot(w.T, DTR[:,LTR==1]) + b
        #cross entropy
        cxe = (1-prior)/nf * numpy.logaddexp(0, -S0*Z0).mean() + prior/nt * numpy.logaddexp(0, -S1*Z1).mean()
        return cxe + 0.5*l* numpy.linalg.norm(w)**2
        pass
    
    return logreg_obj

def logreg(DTR,LTR,DTE,LTE,lamb,prior):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, lamb, prior)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True,iprint=-1)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    nt = DTR[:,LTR==1].shape[0]
    nf = DTR[:,LTR==0].shape[0]
    STE = numpy.dot(_w.T, DTE) + _b - numpy.log(nt/nf) #We unplug the prior to test the application one.
    
    return STE.tolist()

def logreg2(DTR,LTR,DTE,LTE,lamb,prior):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, lamb, prior)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True,iprint=-1)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    nt = DTR[:,LTR==1].shape[0]
    nf = DTR[:,LTR==0].shape[0]
    #STE = numpy.dot(_w.T, DTE) + _b - numpy.log(nt/nf) #We unplug the prior to test the application one.
    
    return _w, _b - numpy.log(nt/nf)

def logreg_cal(DTR,LTR,DTE,LTE,lamb,prior):
    
    logreg_obj = logreg_obj_wrap(DTR, LTR, lamb, prior)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0]+1), approx_grad=True,iprint=-1)
    _w = _v[0:DTR.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b - numpy.log(prior/(1-prior))
    
    return STE.tolist()

def lambdaTuning(D,L,k,gaussianize=0,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx_ = numpy.random.permutation(D.shape[1])
    # _ , DP = compute_PCA(D,7)
    scores = []
    labels = []
    lambs = numpy.logspace(-5, -1, num=30)
    y = []
    priors = [0.5,0.1,0.9]
    for pi in priors:
        scores = []
        labels = []
        minDCFs = []
        idx = idx_
        for l in lambs:
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
                scores.extend(logreg(DTR, LTR, DTE, LTE, l, 0.5))
                idx = numpy.roll(idx,nTest,axis=0)
            print('lambda: ', l)
            minDCFs.append(DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
        y.append(numpy.hstack(minDCFs))
        print(y)
    DCF.plot_DCF(lambs, y, 'lambda', 'LR_lamdaTuning.svg')
        

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return 

def logRegModel(D,L,k, gaussianized=0, seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx_ = numpy.random.permutation(D.shape[1])
    # _ , DP = compute_PCA(D,7)
    scores = []
    labels = []
    priors = [0.5,0.1,0.9]
    for pi_T in priors:
        scores = []
        labels = []
        idx = idx_
        for pi in priors:
            scores = []
            labels = []
            idx = idx_
            for i in range(k):
                #print("Fold :", i+1)
                idxTrain = idx[0:nTrain] 
                idxTest = idx[nTrain:]
                DTR = D[:, idxTrain] 
                DTE = D[:, idxTest]
                if gaussianized==1:
                    DTR = f.gaussianize_features(DTR, DTR)
                    DTE = f.gaussianize_features(DTR, DTE)
                LTR = L[idxTrain] 
                LTE = L[idxTest]
                #Applico PCA
                P=compute_PCA(DTR,7)
                DTR=PCA(P,DTR)
                DTE=PCA(P,DTE)
                ##############
                labels.append(LTE.tolist())
                scores.extend(logreg(DTR, LTR, DTE, LTE, 1e-5, pi_T))
                idx = numpy.roll(idx,nTest,axis=0)
            #if gaussianized == 1:
                #print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' with gaussianized features : ', DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
            #else :
            print('minDCF LR with prior ', pi_T ,' and application ', pi,', ', 1,', ', 1 , ' : ', "%.3f" % DCF.compute_min_DCF(scores, numpy.hstack(labels), pi, 1, 1))
            #if pi == 0.5 and pi_T == 0.5:
                #DCF.plot_minDCF(scores, numpy.hstack(labels))
        

    
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return scores

if __name__ == '__main__':
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    ZD = f.ZNormalization(D)
    s = logRegModel(ZD,L,3)