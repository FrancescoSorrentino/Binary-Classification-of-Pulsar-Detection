# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""

import numpy
import matplotlib
import scipy.special
import matplotlib.pyplot as plt
import DCF
import features 

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

def logpdf_GAU_ND2(X,mu,C) :
    
    res = -0.5*X.shape[0]*numpy.log(2*numpy.pi)
    res += -0.5*numpy.linalg.slogdet(C)[1]
    res += -0.5*((X-mu)*numpy.dot(numpy.linalg.inv(C), (X-mu))).sum(0) #1
    return res

#this function is for the Naive Bayes.
def empirical_diag_cov(D, muc):
    DC = D - muc #class samples centered
    C = (numpy.dot(DC , DC.T))/D.shape[1]
    I = numpy.identity(C.shape[0]) #change it with PCA Hyper parameter
    return C*I

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

#Funzioni aggiunte
def normalized_detection_cost_function (DCF, pi1, cfn, cfp):
    dummy = numpy.array([pi1*cfn, (1-pi1)*cfp])
    index = numpy.argmin (dummy) 
    return DCF/dummy[index]

def minimum_detection_costs (llr, LTE, pi1, cfn, cfp):
    
    sorted_llr = numpy.sort(llr)
    
    NDCF= []
    
    for t in sorted_llr:
        predictions = (llr > t).astype(int)
        
        confMatrix =  confusionMatrix(predictions, LTE, LTE.max()+1)
        uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
        NDCF.append(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    index = numpy.argmin(NDCF)
    
    return NDCF[index]

def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = numpy.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix

def detection_cost_function (M, pi1, cfn, cfp):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    
    return (pi1*cfn*FNR +(1-pi1)*cfp*FPR)

def BestMVG(D,L,k,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1])
    llrTied=[]
    scores=[]
    labels=[]
    priors=[0.5,0.1,0.9]
    for i in range(k):
        #print("Fold :", i+1)
        idxTrain = idx[0:nTrain] 
        idxTest = idx[nTrain:]
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        #Applico PCA
        P=compute_PCA(DTR,7)
        DTR=PCA(P,DTR)
        DTE=PCA(P,DTE)
        #-----
        LTR = L[idxTrain] 
        LTE = L[idxTest]
        labels.extend(LTE)
        llrTied1 = log_TCG(DTR, LTR, DTE, LTE)
        llrTied.extend(llrTied1.tolist())
        idx = numpy.roll(idx,nTest,axis=0)
        
    for p in priors: 
        #compute_min_DCF(scores, labels, pi, Cfn, Cfp)
        print ("Min DCF Application ",p,",1,1 - TIED - raw features", DCF.compute_min_DCF(llrTied, numpy.hstack(labels), p, 1, 1)) #calcolo della DCF del tied
        
        #DCF.plot_minDCF(llrMVG, numpy.hstack(labels))
        #formula generale DCF: copmute_min_DCF(scores,lavels,pi,1,1)
        #scores = llr, labels = labels del training, pi = priors (p), 1 e 1 sono i costi di default

    #print("Avarage error with cross-validation MVG PCA=7 prior: 0.1: ", "%.3f" % (acc*100/k) , "%")
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    return llrTied, numpy.hstack(labels)

def k_fold(D,L,k,gaussianize=0,seed=0):
    nTrain = int(D.shape[1]*(k-1)/k)
    nTest = int(D.shape[1]) - nTrain
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1])
    DP = PCA(D,7)
    #DP=PCA(D, 7, D)
    #mi servono 3 vettori per gli llr
    llrTied=[];
    llrMVG=[];
    llrNaive=[];
    labels=[]
    #acc = 0
    #acc2 = 0
    #acc3 = 0
    #prior da testare
    priors=[0.5,0.1,0.9]
    for p in priors: 
        llrTied=[];
        llrMVG=[];
        llrNaive=[];
        labels=[]
        for i in range(k):
            #print("Fold :", i+1)
            idxTrain = idx[0:nTrain] 
            idxTest = idx[nTrain:]
            DTR = D[:, idxTrain]
            #print("SHAPE DTR",DTR.shape)
            DTE = D[:, idxTest]
            if gaussianize==1:
                DTR = features.gaussianize_features(DTR, DTR)
                DTE = features.gaussianize_features(DTR, DTE)
            #Applico PCA
            #P=compute_PCA(DTR,7)
            #DTR=PCA(P,DTR)
            #DTE=PCA(P,DTE)
            #-----
            LTR = L[idxTrain] 
            LTE = L[idxTest]
            labels.extend(LTE)
            llrMVG1 = log_mvg(DTR, LTR, DTE, LTE)
            llrTied1 = log_TCG(DTR, LTR, DTE, LTE)
            llrNaive1 = log_naive_bayes(DTR, LTR, DTE, LTE)
            llrTied.extend(llrTied1.tolist())
            llrMVG.extend(llrMVG1.tolist())
            llrNaive.extend(llrNaive1.tolist())
            #post training 
            #acc += err
            #acc2 += err2
            #acc3 += err3
            idx = numpy.roll(idx,nTest,axis=0)
        #compute_min_DCF(scores, labels, pi, Cfn, Cfp)
        print ("Min DCF Application ",p,",1,1 - MVG - raw features", DCF.compute_min_DCF(llrMVG, numpy.hstack(labels), p, 1, 1)) #calcolo delle DCF
        print ("Min DCF Application ",p,",1,1 - TIED - raw features", DCF.compute_min_DCF(llrTied, numpy.hstack(labels), p, 1, 1)) #calcolo della DCF del tied
        print ("Min DCF Application ",p,",1,1 - NAIVE - raw features", DCF.compute_min_DCF(llrNaive, numpy.hstack(labels), p, 1, 1)) #dcf naive
        
        #DCF.plot_minDCF(llrMVG, numpy.hstack(labels))
        #formula generale DCF: copmute_min_DCF(scores,lavels,pi,1,1)
        #scores = llr, labels = labels del training, pi = priors (p), 1 e 1 sono i costi di default

    #print("Avarage error with cross-validation MVG PCA=7 prior: 0.1: ", "%.3f" % (acc*100/k) , "%")
    #print("Avarage error with cross-validation Tied PCA=7 prior: 0.1: ", "%.3f" % (acc2*100/k) , "%")
    #print("Avarage error with cross-validation Naive PCA=7 prior: 0.1: ", "%.3f" % (acc3*100/k) , "%")
    #"R" are the train, "E" are evaluations.
    #return acc/k

def tied_covariance(D,mu,L):
    
    Sw = 0 #within class covariance matrix
    for i in range(2):
        Dc = D[:,L==i] #class samples
        DCc = Dc - mu[i] #class samples centered
        Cc = (numpy.dot(DCc , DCc.T))
        Sw += Cc
    Sw= Sw/float(D.shape[1])
    
    return Sw

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

def log_mvg(D,L,DE,LE):
    #i calculate the empirical mean and covariance of the classes.
    mu=[]
    C=[]
    S = []
    for i in range(2):
        Dc = D[:,L==i]
        mu.append(empirical_mean(Dc))
        C.append(empirical_cov(Dc, mu[i]))
    
    for i in range(2):
        ll = logpdf_GAU_ND2(DE, mu[i], C[i])
        
        S.append(ll) #here we keep it as logs.
    #S is a list of array, so we need to transform that list in a matrix
    S = numpy.vstack(S)
    llr=S[1]-S[0] #+ numpy.log(1/2)
# =============================================================================
#     #devo ritornare gli LLR per il k - fold validation
#     #we now multiply it times the Prior probability
#     #and we take the joint distibution.
#     #S[0,:] += numpy.log(1/2)
#     #S[1,:] += numpy.log(1/2)
#     #print(S)
#     SJoint = S
#     #now we compute the marginal density summing over the calsses (row)
#     SMarginal = mrow(scipy.special.logsumexp(SJoint, axis=0))
#     #we calculate the postirior probability.
#     SPost= numpy.exp(SJoint - SMarginal)
#     #we compute the predicted labels using argmax into the rows
#     predicted = numpy.argmax(SPost,0)
#     #we compute the accuracy (we sum the true values)
#     
#     acc = (predicted==LE).sum(0)/DE.shape[1]
#     #we compute the error (1 = 100%)
#     err = 1-acc
#     #print('Model error rate - MVG:', "%.3f" % (err*100), '%')
# =============================================================================
    
    
    return llr

#Tied Covariance Gaussian Classifier.
#Data,Label,Validation Data, Validation Label
def log_TCG(D,L,DE,LE):
    #i calculate the empirical mean and covariance of the classes.
    mu=[]
    S = []
    
    for i in range(2):
        Dc = D[:,L==i]
        mu.append(empirical_mean(Dc))
    
    C = tied_covariance(D, mu, L)
    
    for i in range(2):
        ll = logpdf_GAU_ND2(DE, mu[i], C)
        
        S.append(ll) #here we keep it as logs.
    #S -> lista di log likelihood sui sample -> tanti log likelihood quanti i sample per la DCF
    #S is a list of array, so we need to transform that list in a matrix
    S = numpy.vstack(S)
    llr=S[1]-S[0] #+ numpy.log(1/2) #log likelihood ratio 
# =============================================================================
#     #devo ritornare gli LLR per il k - fold validation
#     #print("llr",llr) #ok
#     #we now multiply it times the Prior probability 
#     #and we take the joint distibution.
#     #S[0,:] += numpy.log(1/2)
#     #S[1,:] += numpy.log(1/2) commentato
#     SJoint = S
#     #now we compute the marginal density summing over the calsses (row)
#     SMarginal = mrow(scipy.special.logsumexp(SJoint, axis=0))
#     #we calculate the postirior probability. -> LLR
#     SPost= numpy.exp(SJoint - SMarginal) #SPost senza prior perché non vengono considerate al momento
#     #we compute the predicted labels using argmax into the rows
#     predicted = numpy.argmax(SPost,0)
#     #we compute the accuracy (we sum the true values)
#     
#     acc = (predicted==LE).sum(0)/DE.shape[1]
#     #we compute the error (1 = 100%)
#     err = 1-acc
#     #print('Model error rate - Tied:', "%.3f" % (err*100), '%')
# =============================================================================
    
    
    return llr

#Naive Bayes classifier
def log_naive_bayes(D,L,DE,LE):
    #i calculate the empirical mean and covariance of the classes.
    mu=[]
    C=[]
    S = []
    for i in range(2):
        Dc = D[:,L==i]
        mu.append(empirical_mean(Dc))
        #we use diagonalized covariance matrix
        C.append(empirical_diag_cov(Dc, mu[i]))
    
    for i in range(2):
        ll = logpdf_GAU_ND2(DE, mu[i], C[i])
        
        S.append(ll) #here we keep it as logs.
    #S is a list of array, so we need to transform that list in a matrix
    S = numpy.vstack(S)
    llr=S[1]-S[0] #+ numpy.log(1/2)
# =============================================================================
#     #devo ritornare gli LLR per il k - fold validation
#     #we now multiply it times the Prior probability
#     #and we take the joint distibution.
#     #S[0,:] += numpy.log(1/2)
#     #S[1,:] += numpy.log(1/2)
#     SJoint = S
#     #now we compute the marginal density summing over the calsses (row)
#     SMarginal = mrow(scipy.special.logsumexp(SJoint, axis=0))
#     #we calculate the postirior probability.
#     SPost= numpy.exp(SJoint - SMarginal)
#     #we compute the predicted labels using argmax into the rows
#     predicted = numpy.argmax(SPost,0)
#     #we compute the accuracy (we sum the true values)
#     
#     acc = (predicted==LE).sum(0)/DE.shape[1]
#     #we compute the error (1 = 100%)
#     err = 1-acc
#     #print('Model error rate - Naive Bayes:', "%.3f" % (err*100), '%')
# =============================================================================
    
    
    return llr

if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    #RAW DATA
    D, L = load('../Train.txt')
    DE, LE = load('../Test.txt')
    #--------------------
    D=features.ZNormalization(D)
    #----------------------
    acc = k_fold(D, L, 3)