# -*- coding: utf-8 -*-
"""
Authors: Francesco Sorrentino, Francesco Di Gangi
"""

import numpy
import scipy.special
import pylab
import matplotlib.pyplot as plt

def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -numpy.log(pi*Cfn) + numpy.log((1-pi)*Cfp)
    P = scores > th
    return numpy.int32(P)

def compute_conf_matrix_binary(Pred, Labels):
    C = numpy.zeros((2,2))
    C[0,0] = ((Pred == 0) * (Labels == 0)).sum()
    C[0,1] = ((Pred == 0) * (Labels == 1)).sum()
    C[1,0] = ((Pred == 1) * (Labels == 0)).sum()
    C[1,1] = ((Pred == 1) * (Labels == 1)).sum()
    return C

def compute_emp_Bayes_binary(CM, pi, Cfn, Cfp):
    fnr = CM[0,1] / (CM[0,1] + CM[1,1])
    fpr = CM[1,0] / (CM[0,0] + CM[1,0])
    return pi * Cfn * fnr + (1-pi) * Cfp * fpr

def compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp): #DCF
    empBayes = compute_emp_Bayes_binary(CM, pi, Cfn, Cfp) 
    return empBayes / min(pi*Cfn, (1-pi)*Cfp)

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_normalized_emp_Bayes(CM, pi, Cfn, Cfp)

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t=numpy.sort(t)
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return numpy.array(dcfList).min()

def bayes_error_plot(pArray, scores, labels, minCost=False):
    y=[]
    for p in pArray:
        pi = 1.0/ (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return numpy.array(y)

def compute_rates(CM, pi, Cfn, Cfp):
    fnr = CM[0,1] / (CM[0,1] + CM[1,1])
    tpr = 1- fnr
    fpr = CM[1,0] / (CM[0,0] + CM[1,0])
    return fnr, fpr

def compute_rates2(CM, pi, Cfn, Cfp):
    fnr = CM[0,1] / (CM[0,1] + CM[1,1])
    tpr = 1- fnr
    fpr = CM[1,0] / (CM[0,0] + CM[1,0])
    return tpr, fpr

def compute_DET(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_rates(CM, pi, Cfn, Cfp)

def compute_ROC(scores, labels, pi, Cfn, Cfp, th=None):
    Pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    CM = compute_conf_matrix_binary(Pred, labels)
    return compute_rates2(CM, pi, Cfn, Cfp)

def DET(pArray, scores, labels):
    y=[]
    x=[]
    t = numpy.array(scores)
    t=numpy.sort(t)
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    for _th in t:
        y1,x1 = compute_DET(scores, labels, 0.5, 1, 1, th=_th)
        y.append(y1)
        x.append(x1)
    return numpy.array(y), numpy.array(x)

def ROC(pArray, scores, labels):
    y=[]
    x=[]
    t = numpy.array(scores)
    t=numpy.sort(t)
    numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    for _th in t:
        y1,x1 = compute_ROC(scores, labels, 0.5, 1, 1, th=_th)
        y.append(y1)
        x.append(x1)
    return numpy.array(y), numpy.array(x)

def plot_DET(svm,mvg,fusion,labels,labels2,name):
    p = numpy.linspace(-3,3,21)
    y,x = DET(p,svm,labels)
    y1,x1 = DET(p,mvg,labels)
    y2,x2 = DET(p,fusion,labels2)
    pylab.plot(x, y, label='SVM', color='red' )
    pylab.plot(x1, y1, label='MVG', color='b')
    pylab.plot(x2, y2, label='Fusion', color='g')
    plt.legend()
# =============================================================================
#     plt.ylim([1,30])
#     plt.xlim([1,95])
# =============================================================================
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('FPR')
    plt.ylabel("FNR")
    plt.grid(linestyle=':')
    plt.savefig(name)
    pylab.show()
    
def plot_ROC(svm,mvg,fusion,labels,labels2,name):
    p = numpy.linspace(-3,3,21)
    y,x = ROC(p,svm,labels)
    y1,x1 = ROC(p,mvg,labels)
    y2,x2 = ROC(p,fusion,labels2)
    pylab.plot(x, y, label='SVM', color='red' )
    pylab.plot(x1, y1, label='MVG', color='b')
    pylab.plot(x2, y2, label='Fusion', color='g')
    plt.legend()
# =============================================================================
#     plt.ylim([1,30])
#     plt.xlim([1,95])
# =============================================================================
    plt.xlabel('FPR')
    plt.ylabel("TPR")
    plt.grid(linestyle=':')
    plt.savefig(name)
    pylab.show()

def plot_minDCF(scores,labels,name):
    p = numpy.linspace(-3,3,21)
    pylab.plot(p, bayes_error_plot(p, scores, labels), label='actDCF', color='red' )
    pylab.plot(p, bayes_error_plot(p, scores, labels, minCost="True"), label='minDCF', color='b')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.savefig(name)
    pylab.show()
    
def plot_minDCF_best(mvgsc, mvgsc1,svmsc,svmsc1,labels,name):
    p = numpy.linspace(-3,3,21)
    pylab.plot(p, bayes_error_plot(p, svmsc1, labels), label='SVM - actDCF', color='red' )
    pylab.plot(p, bayes_error_plot(p, svmsc, labels, minCost="True"),label='SVM - minDCF', linestyle='dashed' , color='red')
    pylab.plot(p, bayes_error_plot(p, mvgsc1, labels),label='MVG - actDCF', color='b' )
    pylab.plot(p, bayes_error_plot(p, mvgsc, labels, minCost="True"),label='MVG - minDCF', linestyle='dashed' ,color='b')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.savefig(name)
    pylab.show()
    
def plot_minDCF_final(mvgsc, svmsc, fussc, labels, labels2, name):
    p = numpy.linspace(-3,3,21)
    pylab.plot(p, bayes_error_plot(p, svmsc, labels), label='SVM (cal) - actDCF', color='red' )
    pylab.plot(p, bayes_error_plot(p, mvgsc, labels, ), label='MVG (cal) - actDCF', color='b')
    pylab.plot(p, bayes_error_plot(p, fussc, labels2), label='Fusion - actDCF', color='g' )
    pylab.plot(p, bayes_error_plot(p, fussc, labels2, minCost="True"),label='Fusion - minDCF', linestyle='dashed' ,color='g')
    plt.legend()
    plt.ylim([0, 0.9])
    plt.xlim([-3, 3])
    plt.savefig(name)
    pylab.show()    
    
def plot_DCF(x, y, xlabel, name,base=10):
    plt.figure()
    plt.plot(x, y[0], label='min DCF prior=0.5', color='b')
    plt.plot(x, y[1], label='min DCF prior=0.1', color='r')
    plt.plot(x, y[2], label='min DCF prior=0.9', color='g')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend(["min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig(name)
    return

def plot_DCF_gamma(x, y, xlabel, name,base=10):
    plt.figure()
    plt.plot(x, y[0], label='min DCF γ= -2', color='orange')
    plt.plot(x, y[1], label='min DCF γ= -1', color='r')
    plt.plot(x, y[2], label='min DCF γ= 0', color='y')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend(["min DCF γ= -2", "min DCF γ= -1", "min DCF γ= 0"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig(name)
    return

def plot_DCF_GMM(x,yz,yg,name):
    fig, ((ax1)) = plt.subplots(1, 1, constrained_layout = True)
    width = 0.25
    ind = numpy.arange(len(x)) 
    b3n = ax1.bar(ind+1, yg, width, color = 'r')
    b3g = ax1.bar(ind+width+1, yz, width, color='g')
    ax1.legend((b3n, b3g), ('Gaussian', 'Z-Normalization'))
    ax1.title.set_text(name)
    plt.savefig(name+".svg",format="svg")