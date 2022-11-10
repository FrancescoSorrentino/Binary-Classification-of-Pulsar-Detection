# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""

import numpy
import matplotlib
import scipy.special
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

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

def plot_hist(D, L, T):
    
    #This takes all the columns, for each one index that is equal
    #In the Label vector L, with L == n !!!!
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    
    
    hPul = {
        0 : 'Mean of the integrated profile',
        1 : 'Standard deviation of the integrated profile',
        2 : 'Excess kurtosis of the integrated profile',
        3 : 'Skewness of the integrated profile',
        4 : 'Mean of the DM-SNR curve',
        5 : 'Standard deviation of the DM-SNR curve',
        6 : 'Excess kurtosis of the DM-SNR curve',
        7 : 'Skewness of the DM-SNR curve',
        }
    
    for dIdx in range(8):
        plt.figure()
        plt.xlabel(hPul[dIdx])
        plt.hist(D0[dIdx, :], bins = 35, density = True, alpha = 0.4, color='blue', label = 'RFI/Noises')
        plt.hist(D1[dIdx, :], bins = 35, density = True, alpha = 0.4, color='red', label = 'Pulsar')
       
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        if T == 0 :
           plt.savefig('plot_f/hist_%d.svg' % dIdx)
        if T == 1:
            plt.savefig('plot_gf/ghist_%d.svg' % dIdx)
        if T == 2:
            plt.savefig('plot_zf/zhist_%d.svg' % dIdx)
        
    plt.show()
    


def gaussianize_features(D,DE):
    P = []
    
    for dIdx in range(8):
       DT = mcol(DE[dIdx,:])
       mean = numpy.mean(D[dIdx,:])
       var = numpy.var(D[dIdx,:])
       X =  D[dIdx,:] < DT
       R = (X.sum(1) + 1)/(D.shape[1] + 2)
       P.append(norm.ppf(R))
                         
    return numpy.vstack(P)

def plot_feature_corr(D,L):
    plt.figure()
    plt.title('Correlation matrix')
    ax = sns.heatmap(numpy.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrix.svg')
    plt.figure()
    plt.title('Correlation matrix with absolute values')
    ax = sns.heatmap(abs(numpy.corrcoef(D)), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrixAvs.svg')
    plt.figure()
    plt.title('Correlation matrix false pulsar')
    ax1 = sns.heatmap(numpy.corrcoef(D[:,L==0]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrixF.svg')
    plt.figure()
    plt.title('Correlation matrix false pulsar with absolute values')
    ax1 = sns.heatmap(abs(numpy.corrcoef(D[:,L==0])), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrixFAbs.svg')
    plt.figure()
    plt.title('Correlation matrix true pulsar')
    ax2 = sns.heatmap(numpy.corrcoef(D[:,L==1]), linewidth=0.2, cmap="Reds" , square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrixT.svg')
    plt.figure()
    plt.title('Correlation matrix true pulsar with absolute values')
    ax2 = sns.heatmap(abs(numpy.corrcoef(D[:,L==1])), linewidth=0.2, cmap="Reds" , square=True, cbar=False)
    plt.savefig('plot_heatmap/CorrMatrixTAbs.svg')
    plt.tight_layout()
    plt.show()

def ZNormalization(D, mean=None, standardDeviation=None):
    if (mean is None and standardDeviation is None):
        mean = D.mean(axis=1)
        standardDeviation = D.std(axis=1)
    ZD = (D-mcol(mean))/mcol(standardDeviation)
    return ZD

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

if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('../Train.txt')
    DZ = ZNormalization(D)
    DG = gaussianize_features(DZ,DZ)
    #plot_hist(D,L,0)
    #plot_hist(DZ,L,2)
    #plot_hist(DG,L,1)
    #plot_hist(DG1,L,1)
    plot_feature_corr(DZ,L)
