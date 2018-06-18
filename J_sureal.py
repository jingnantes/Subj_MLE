#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:18:29 2018

@author: jingli
"""

from __future__ import division
import numpy as np
#from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.optimize import SR1
from scipy.optimize import Bounds
from scipy.optimize import BFGS
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
import csv
import pandas as pd
from numpy import genfromtxt

__copyright__ = "Copyright 2018, Jing LI, LS2N Lab, University of Nantes"
__license__ = "Apache, Version 2.0"

#global hrc_num
#global src_num
#global obs_num
#global data
#src_num = 9
#hrc_num = 16
#obs_num = 24
#pvs_num = src_num*hrc_num

""" likelihood function and its first-order derivatives"""

def Lfunction(x):
    """ the log likelihood function for MLE subjective model"""
    
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    
    res = 0
    pvs = 0
    
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            for obs in range(0,obs_num):
                if data[src,hrc,obs]!=0:
                    res = res + np.log(vs[obs]**2+ae[src]**2)+np.power((data[src,hrc,obs]-xe[pvs]-bs[obs]),2)/(vs[obs]**2+ae[src]**2)
            pvs = pvs+1    
    return res

def Lfunction_det(x):
    """ the log likelihood function considering hessian matrix semi-definite for MLE subjective model"""
    
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    u = x[-1]
    
    
    res = 0
    pvs = 0
    
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            
            for obs in range(0,obs_num):
                if data[src,hrc,obs]!=0:
                    res = res + np.log(vs[obs]**2+ae[src]**2)+np.power((data[src,hrc,obs]-xe[pvs]-bs[obs]),2)/(vs[obs]**2+ae[src]**2)
            pvs = pvs+1        
    return res + u*np.log(np.linalg.det(Hessian_L(x[0:-1])))
        
def L_dev(x):
    # consider that vs and ae are variables
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    
    dxe = np.zeros_like(xe)
    dbs = np.zeros_like(bs)
    dvs = np.zeros_like(vs)
    dae = np.zeros_like(ae)
    
    xx = np.zeros((src_num,hrc_num))
    tmp = -1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            tmp = tmp+1
            xx[src,hrc] = xe[tmp]
            
    
    e=0
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            init = 0
            for s in range(0,obs_num):
                if data[src,hrc,s]!=0:
                    init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
            dxe[e] = init
            e=e+1
            
    for s in range(0,obs_num):
        init = 0
        init2 = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                if data[src,hrc,s]!=0:
                    init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
                    init2 = init2 + 2*vs[s]/(vs[s]**2+ae[src]**2) -2*vs[s]*np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        
        dbs[s] = init
        dvs[s] = init2
           
            
    for src in range(0,src_num):
        init = 0
        for hrc in range(0,hrc_num):
            for s in range(0,obs_num):
                if data[src,hrc,s]!=0:                 
                    init = init + 2*ae[src]/(vs[s]**2+ae[src]**2) -2*ae[src]*np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        dae[src] = init 
            
    der = np.r_[dxe,dbs,dvs,dae]
    return der

def L_dev2(x):
    # consider that vs^2 and ae^2 are variables
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    
    dxe = np.zeros_like(xe)
    dbs = np.zeros_like(bs)
    dvs = np.zeros_like(vs)
    dae = np.zeros_like(ae)
    
    xx = np.zeros((src_num,hrc_num))
    tmp = -1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            tmp = tmp+1
            xx[src,hrc] = xe[tmp]
            
    
    e=0
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            init = 0
            
            for s in range(0,obs_num):
                if data[src,hrc,s]!=0:
                    init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
            dxe[e] = init
            e=e+1
            
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                if data[src,hrc,s]!=0:
                    init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
        dbs[s] = init  
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                if data[src,hrc,s]!=0:
                    init = init + 1/(vs[s]**2+ae[src]**2) -np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        dvs[s] = init  
            
    for src in range(0,src_num):
        init = 0
        for hrc in range(0,hrc_num):
            for s in range(0,obs_num): 
                if data[src,hrc,s]!=0:                
                    init = init + 1/(vs[s]**2+ae[src]**2) -np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        dae[src] = init 
            
    der = np.r_[dxe,dbs,dvs,dae]
    return der

def Hessian_L(x):
    # consider vs^2 and ae^2 as variables
    
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    
    xx = np.zeros((src_num,hrc_num))
    tmp = -1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            tmp = tmp+1
            xx[src,hrc] = xe[tmp]
    
    dxe2 = np.zeros((len(xe),len(xe)))
    dxebs= np.zeros((len(xe),len(bs)))
    dxevs= np.zeros((len(xe),len(vs)))
    dxeae= np.zeros((len(xe),len(ae)))
    
    dbsxe  = np.zeros((len(bs),len(xe)))
    dbs2  = np.zeros((len(bs),len(bs)))
    dbsvs = np.zeros((len(bs),len(vs)))
    dbsae = np.zeros((len(bs),len(ae)))
    
    dvsxe= np.zeros((len(vs),len(xe)))
    dvsbs= np.zeros((len(vs),len(bs)))
    dvs2= np.zeros((len(vs),len(vs)))
    dvsae= np.zeros((len(vs),len(ae)))
    
    daexe= np.zeros((len(ae),len(xe)))
    daebs= np.zeros((len(ae),len(bs)))
    daevs= np.zeros((len(ae),len(vs)))
    dae2= np.zeros((len(ae),len(ae)))
    
    ## xe
    pvs = 0
    for src in range(src_num):
        for hrc in range(hrc_num):
            init = 0
            
            for obs in range(obs_num):
                if data[src,hrc,obs]!=0:
                    init =init + 2/(vs[obs]**2+ae[src]**2)
                    dxebs[pvs,obs] = 2/(vs[obs]**2+ae[src]**2)
                    dxevs[pvs,obs] = 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
            dxe2[pvs,pvs] = init
            pvs = pvs+1
            
    pvs = 0            
    for src in range(src_num):
        for hrc in range(hrc_num):
            init = 0
            for obs in range(obs_num):
                if data[src,hrc,obs]!=0:
                    init = init + 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
            dxeae[pvs,src] = init
            pvs = pvs+1
    ## bs   
    dbsxe = dxebs.T
    for obs in range(obs_num):
        dbs2[obs,obs] = 2*np.sum(1/(vs[obs]**2+ae**2))
        init = 0
        
        for src in range(src_num):
            init2 = 0
            for hrc in range(hrc_num):
                if data[src,hrc,obs]!=0:
                    init = init + 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
                    init2 = init2 + 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
            dbsae[obs,src] = init2
        dbsvs[obs,obs] = init
    
    ## vs
    dvsxe = dxevs.T
    dvsbs = dbsvs.T
    for obs in range(obs_num):
        init = 0
        for src in range(src_num):
            init2 = 0
            for hrc in range(hrc_num):
                if data[src,hrc,obs]!=0:
                    init = init -1/(vs[obs]**2+ae[src]**2)**2+2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])**2/(vs[obs]**2+ae[src]**2)**3
                    init2 = init2 -1/(vs[obs]**2+ae[src]**2)**2+2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])**2/(vs[obs]**2+ae[src]**2)**3
            dvsae[obs,src] = init2          
        dvs2[obs,obs] = init
    
    ## ae
    daexe = dxeae.T
    daebs = dbsae.T
    daevs = dvsae.T
    for src in range(src_num):
        init = 0
        for hrc in range(hrc_num):
            for obs in range(obs_num):
                if data[src,hrc,obs]!=0:
                    init = init -1/(vs[obs]**2+ae[src]**2)**2+2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])**2/(vs[obs]**2+ae[src]**2)**3
        dae2[src,src]= init   
          
    xe_row = np.c_[dxe2,dxebs,dxevs,dxeae]
    bs_row = np.c_[dbsxe,dbs2,dbsvs,dbsae]
    vs_row = np.c_[dvsxe,dvsbs,dvs2,dvsae]
    ae_row = np.c_[daexe,daebs,daevs,dae2]
     
    hess = np.r_[xe_row,bs_row,vs_row,ae_row]
    return hess         
    
def EIG_L(x):
    hess = Hessian_L(x)
    return np.linalg.eigvals(hess)    

def cons_f(x):
    return np.linalg.eigvals(Hessian_L(x))

def CI_parameters(x):
    return 1.96/np.sqrt(np.diag(Hessian_L(x)))

def HRC_MOS(data,src_num,hrc_num,obs_num):
    hrcmos = np.zeros((hrc_num,src_num*obs_num))
    for hrc in range(0,hrc_num):
        n = -1
        for src in range(0,src_num):
            for obs in range(0,obs_num):
                n=n+1;
                if data[src,hrc,obs]==0:
                    hrcmos[hrc,n] = 'nan'
                else:
                    hrcmos[hrc,n]=data[src,hrc,obs]   
    return hrcmos

def SRC_MOS(data,src_num,hrc_num,obs_num):
    srcmos = np.zeros((src_num,hrc_num*obs_num))
    for src in range(0,src_num):
        n = -1
        for hrc in range(0,hrc_num):
            for obs in range(0,obs_num):
                n=n+1;
                if data[src,hrc,obs]==0:
                    srcmos[src,n]='nan'
                else:
                    srcmos[src,n]=data[src,hrc,obs]   
    return srcmos

def Suj_MOS(data,src_num,hrc_num,obs_num):
    sujmos = np.zeros((obs_num,src_num*hrc_num))
    for obs in range(0,obs_num):
        n = -1
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                n=n+1;
#                if data[src,hrc,obs]==0:
#                    sujmos[obs,n] = 'nan'
#                else:
#                    sujmos[obs,n]=data[src,hrc,obs]
                sujmos[obs,n]=data[src,hrc,obs]
    return sujmos

def MLE_process(data,src_num,hrc_num,pvs_num,obs_num,opt_method):
    """ Initialization value for xe,bs,vs, ae """
    
    
    xe0 = np.nanmean(Suj_MOS(data,src_num,hrc_num,obs_num),0) # calculate column mean
    bs0 = np.zeros(obs_num)
    vs0 = np.nanstd(Suj_MOS(data,src_num,hrc_num,obs_num),axis=1,ddof=1)
    ae0 = np.nanstd(SRC_MOS(data,src_num,hrc_num,obs_num),axis=1,ddof=1)

    xinput = np.r_[xe0,bs0,vs0,ae0]

    """ set lower and upper bounds for each parameter """
    tempstd = np.nanstd(SRC_MOS(data,src_num,hrc_num,obs_num),axis=1,ddof=1)
    srcusestd = np.nanmax(tempstd)

    tempstd = np.nanstd(Suj_MOS(data,src_num,hrc_num,obs_num),axis=1,ddof=1)
    obsusestd = np.nanmax(tempstd)

    xelb = np.zeros(pvs_num)
    xeub = 6*np.ones(pvs_num)
    bslb = -2*np.ones(obs_num)
    bsub = 2*np.ones(obs_num)
    vslb = 0.001*np.ones(obs_num)
    vsub = obsusestd*np.ones(obs_num)
    aelb = 0.001*np.ones(src_num)
    aeub = srcusestd*np.ones(src_num)

    lb = np.r_[xelb,bslb,vslb,aelb]
    ub = np.r_[xeub,bsub,vsub,aeub]

    bounds = Bounds(lb, ub)
    
    cons = np.c_[np.zeros((1,pvs_num)),np.ones((1,obs_num)),np.zeros((1,obs_num+src_num))]
    linear_constraint = LinearConstraint(cons,[0],[0])
    #nonlinear_constraint = NonlinearConstraint(cons_f, 0, np.inf, jac='2-point', hess=BFGS())

    """using minimize method to find MLE estimates """
    ## nelder-mead method is worse than BFGS
    if opt_method == 'Nelder-Mead':
        res = minimize(Lfunction, xinput, method='nelder-mead',options={'xtol':1e-8, 'disp':True})
    elif opt_method == 'BFGS':
        res = minimize(Lfunction, xinput, method='BFGS',jac = L_dev, options={'disp':True})
    elif opt_method == 'Newton-CG': 
        #res = minimize(Lfunction, xinput, method='Newton-CG',jac = L_dev2,options={'disp':True})
        res = minimize(Lfunction, xinput, method='Newton-CG',jac = L_dev, hess = Hessian_L, options={'disp':True})
    elif opt_method == 'trust-constr':
        res = minimize(Lfunction, xinput, method='trust-constr',  jac=L_dev, hess = Hessian_L, constraints=[linear_constraint], options={'verbose': 1}, bounds=bounds)
    else:   
        #res = minimize(Lfunction, xinput, method='SLSQP', jac = '2-point', options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
        res = minimize(Lfunction, xinput, method='SLSQP', jac = L_dev, options={'ftol': 1e-9, 'disp': True}, bounds=bounds)

        """consider using constraints hessian matrix semi-definite """
        #res = minimize(Lfunction_det, np.r_[xinput,0.001], method='nelder-mead',options={'xtol':1e-8, 'disp':True})

    print res
    ci = CI_parameters(res.x)
    return res,ci

def Draw_figure(parameter,para_ci):
    
    (xee,bse,vse,aee) = parameter
    (xeci,bsci,vsci,aeci) = para_ci
    
    plt.figure()
    plt.errorbar(np.arange(len(xee)), xee, yerr=xeci, fmt='o')
    plt.title('HRC quality')
    plt.xlabel('HRC No.')
    plt.ylabel('value')
    

    plt.figure()
    plt.errorbar(np.arange(len(bse)), bse, yerr=bsci, fmt='o')
    plt.title('observer bias')
    plt.xlabel('Observer No.')
    plt.ylabel('value')
    
    plt.figure()
    plt.errorbar(np.arange(len(vse)), vse, yerr=vsci, fmt='o')
    plt.title('inconsistency')
    plt.xlabel('Observer No.')
    plt.ylabel('value')

    plt.figure()
    #plt.errorbar(np.arange(len(aee)), aee, yerr=aeci, fmt='o')
    plt.scatter(np.arange(len(aee)), aee, marker=r'$\clubsuit$')
    plt.title('content ambiguity')
    plt.xlabel('SRC No.')
    plt.ylabel('value')
    
    plt.figure()
    plt.errorbar(np.arange(len(aee)), aee, yerr=aeci, fmt='o')
    #plt.scatter(np.arange(len(aee)), aee, marker=r'$\clubsuit$')
    plt.title('content ambiguity')
    plt.xlabel('SRC No.')
    plt.ylabel('value')

def Structure_res(res,ci,src_num,pvs_num,obs_num):
    
    xee = res.x[range(0,pvs_num)]
    bse = res.x[range(pvs_num,pvs_num+obs_num)]
    vse = res.x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    aee = res.x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]    
    
    xeci = ci[range(0,pvs_num)]
    bsci = ci[range(pvs_num,pvs_num+obs_num)]
    vsci = ci[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    aeci = ci[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
       
    parameter = (xee,bse,vse,aee)
    para_ci = (xeci,bsci,vsci,aeci)
    return parameter, para_ci

def Outlier_analysis(parameter,para_ci):
    (xee,bse,vse,aee) = parameter
    (xeci,bsci,vsci,aeci) = para_ci
    
    plt.figure()
    plt.scatter(bse,vse,marker=r'$\clubsuit$')
    plt.title('scatter plot of subjects')
    plt.xlabel('Subject bias')
    plt.ylabel('Subject inconsistency')
    
    plt.figure()
    plt.errorbar(bse,vse,xerr = bsci, yerr = vsci, fmt = 'o')
    plt.title('subject analysis')
    plt.xlabel('Subject bias')
    plt.ylabel('Subject inconsistency')

def ReadFile(file_name):
    data = np.load(file_name)
    return data

def Synthesis_data(src_num,hrc_num,pvs_num,obs_num):
    data = np.zeros((src_num,hrc_num,obs_num))
    ## do not change this part
    """ true score"""
    xe = np.random.uniform(1,5,pvs_num)

    """ observer bias """
    bs = np.random.uniform(-2,2,obs_num)
    #bs = np.random.normal(0, 1, obs_num)

    """ observer inconsistency """
    vs = np.random.uniform(0.2,0.8,obs_num)


    """ content ambicuity """
    ae = np.random.uniform(0.1,0.5,src_num)


    """ synthesized observed score """

    xes =[[0 for i in range(pvs_num)]for i in range(obs_num)]
    xes = np.array(xes)

#    pvs = -1
#    src_idx = np.zeros(pvs_num)
#    for src in range(0,src_num):
#        for hrc in range(0,hrc_num):
#            pvs = pvs+1
#            src_idx[pvs] = src
               
   # np.random.seed(0)
    pvs = -1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            pvs = pvs+1
            for obs in range(0,obs_num):
                mu = xe[pvs]+bs[obs]
                sigma = np.sqrt(np.power(vs[obs],2)+np.power(ae[src],2))
                xes[obs,pvs] = round(np.random.normal(mu, sigma, 1))
                #xes[obs,pvs] = Clipping_data(temp)
                data[src,hrc,obs] = xes[obs,pvs]
    # finish the synthesis process
    return xe,bs,vs,ae,data

def Clipping_data(x):
    if x > 5:
        x = 5
    if x < 1:
        x = 1
    return x

def Synthesis_performance(gt,parameter):
    (xe,bs,vs,ae) = gt
    (xee,bse,vse,aee) = parameter
    plcc_xe = pearsonr(xe,xee)
    plcc_bs = pearsonr(bs,bse)
    plcc_vs = pearsonr(vs,vse)
    plcc_ae = pearsonr(ae,aee)
    
    plcc = (plcc_xe,plcc_bs,plcc_vs,plcc_ae)
    
    rmse_xe = np.sqrt(np.mean((xe-xee)**2))
    rmse_bs = np.sqrt(np.mean((bs-bse)**2))
    rmse_vs = np.sqrt(np.mean((vs-vse)**2))
    rmse_ae = np.sqrt(np.mean((ae-aee)**2))
    
    rmse = (rmse_xe,rmse_bs,rmse_vs,rmse_ae)
    
    plt.figure()
    plt.scatter(xe,xee,marker=r'$\clubsuit$')
    plt.xlabel("ground truth score, xe")
    plt.ylabel("estimated xee")
    plt.show()

    plt.scatter(bs,bse,marker=r'$\clubsuit$')
    plt.xlabel("ground truth observer bias, bs")
    plt.ylabel("estimated bse")
    plt.show()

    plt.scatter(vs,vse,marker=r'$\clubsuit$')
    plt.xlabel("ground truth observer inconsistency, vs")
    plt.ylabel("estimated vse")
    plt.show()

    plt.scatter(ae,aee,marker=r'$\clubsuit$')
    plt.xlabel("ground truth content ambiguity, ae")
    plt.ylabel("estimated aee")
    plt.show()
    
    
    return plcc,rmse

def CI_performance_MLE(gt,parameter,para_ci,who):
    ## who means 'xe','bs','vs','ae', which one do you want to check
    
    (xe,bs,vs,ae) = gt
    (xee,bse,vse,aee) = parameter
    (xeci,bsci,vsci,aeci) = para_ci
    
    
    if who == 'xe':
        res = np.zeros_like(xe)
        for i in range(len(xe)):
            if xe[i] >= xee[i]-xeci[i] and xe[i] <= xee[i]+xeci[i]:
                res[i] = 1
            else:
                res[i] = 0
    elif who == 'bs':
        res = np.zeros_like(bs)
        for i in range(len(bs)):
            if bs[i] >= bse[i]-bsci[i] and bs[i] <= bse[i]+bsci[i]:
                res[i] = 1
            else:
                res[i] = 0
    elif who == 'vs':
        res = np.zeros_like(vs)
        for i in range(len(vs)):
            if vs[i] >= vse[i]-vsci[i] and vs[i] <= vse[i]+vsci[i]:
                res[i] = 1
            else:
                res[i] = 0
    elif who == 'ae':
        res = np.zeros_like(ae)
        for i in range(len(ae)):
            if ae[i] >= aee[i]-aeci[i] and ae[i] <= aee[i]+aeci[i]:
                res[i] = 1
            else:
                res[i] = 0
    else:
        print("the fourth parameter is not correct!")
        
    return res

def CI_performance_MOS(gt,data):
    ## who means 'xe','bs','vs','ae', which one do you want to check
    
    (xe,bs,vs,ae) = gt
    mos = Suj_MOS(data,src_num,hrc_num,obs_num)
    mos_mean = np.mean(mos,0)
    mos_std = np.std(mos,0)
    mos_ci = 1.96*mos_std/np.sqrt(obs_num)
    
    res = np.zeros_like(xe)
    for i in range(len(xe)):
        if xe[i] >= mos_mean[i]-mos_ci[i] and xe[i] <= mos_mean[i]+mos_ci[i]:
                res[i] = 1
        else:
                res[i] = 0
    
    return res

def Write_file_csv(row_vector,filename):
    f1 = open(filename,'a')
    with f1:
        writer = csv.writer(f1,delimiter=',', lineterminator='\n')
        #writer.writerow(np.asarray(vector))
        writer.writerow(row_vector)
    f1.close()
    #
    #f1.write(str(vector)+'\n')

     
def Read_file_csv(filename):
    my_data = genfromtxt(filename,delimiter=',')
    return my_data    
            
#def Write_file_excel(vector,filename):
#    file = xlwt.Workbook(encoding = 'utf-8')
#    table = file.add_sheet('data')
#    for i in len(vector):
#        table.write()

def Analysis_Compare_CI_performance(filename_mle,filename_mos):
    mle_ci = Read_file_csv(filename_mle)
    mos_ci = Read_file_csv(filename_mos)
    
    mle_ci_pvs = np.mean(mle_ci,1)
    mos_ci_pvs = np.mean(mos_ci,1)
    
    mle_ci_iteration = np.mean(mle_ci,0)
    mos_ci_iteration = np.mean(mos_ci,0)
    
    mle_ci_all = np.mean(mle_ci_pvs)
    mos_ci_all = np.mean(mos_ci_pvs)
    return mle_ci_pvs, mle_ci_iteration, mle_ci_all, mos_ci_pvs,mos_ci_iteration,mos_ci_all

def Bootstrapping_observations(mu,sigma):
    iteration = 100
    bos = np.zeros((iteration,len(mu)))
    for i in range(iteration):
        for j in range(len(mu)):
            bos[i,j] = np.random.normal(mu[j], sigma[j],1)
    bos_mean = np.mean(bos,0)
    bos_std = np.std(bos,0)
    bos_ci = 1.96*bos_std/np.sqrt(iteration)
    return bos_mean, bos_std, bos_ci
        
def Boost_CI_performance(gt,ci_lower,ci_upper):
    res = np.zeros_like(gt)
    for i in range(len(gt)):
        if gt[i] >= ci_lower[i] and gt[i] <= ci_upper[i]:
            res[i] = 1
        else:
            res[i] = 0        
    return res

def Bootstrapping_analysis(gt,data):
    (xe,bs,vs,ae) = gt
    opt_method = 'SLSQP'
    mle_res,mle_ci = MLE_process(data,src_num,hrc_num,pvs_num,obs_num,opt_method)
    mle_parameter,mle_para_ci = Structure_res(mle_res,mle_ci,src_num,pvs_num,obs_num)
    (xe_mle,bs_mle,vs_mle,ae_mle) = mle_parameter
    
    re_sampling_time = 100
    xee = np.zeros((re_sampling_time,len(xe)))
    boot_data = np.zeros_like(data)
    ## resampling from data
    for i in range(re_sampling_time):
        for obs in range(obs_num):
            boot_data[:,:,obs] = data[:,:,np.random.randint(0,obs_num)]
        
        res,ci = MLE_process(boot_data,src_num,hrc_num,pvs_num,obs_num,opt_method)
        parameter,para_ci = Structure_res(res,ci,src_num,pvs_num,obs_num)
        (xee[i,:],bse,vse,aee) = parameter    
    
    delta_star = xee-xe
    np.savez('ci_boost_temp_slsqp',delta_star,xe,xee)
    
    ci_lower = np.zeros_like(xe)
    ci_upper = np.zeros_like(xe)
    for i in range(len(xe)):
        nsort = np.sort(delta_star[:,i])
        ci_lower[i] = xe_mle[i] - nsort[98]
        ci_upper[i] = xe_mle[i] - nsort[2]
    
    res = [ci_lower,ci_upper]
    
    
    res_perform = Boost_CI_performance(xe,ci_lower,ci_upper)
    
#    res_ite_mean = np.mean(res,0)
#    res_pvs_mean = np.mean(res,1)
#    
#    Draw_res_CI_performance(res_ite_mean,'mean_ite','Bootstrapping')
#    Draw_res_CI_performance(res_pvs_mean,'mean_ite','Bootstrapping')
    
    np.savez('ci_boost_slsqp_all',delta_star,xe,xee,res,res_perform)
    return res_perform
    
def Draw_res_CI_performance(ci,xlabel,title):
    plt.figure()
    plt.hist(ci)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()    

def Run_synthesis_experiment(opt_method):
    src_num = 9
    hrc_num = 16
    obs_num = 24
    pvs_num = src_num*hrc_num
    global data
    xe,bs,vs,ae,data = Synthesis_data(src_num,hrc_num,pvs_num,obs_num)
    
    gt = (xe,bs,vs,ae)
    res,ci = MLE_process(data,src_num,hrc_num,pvs_num,obs_num,opt_method)
    parameter,para_ci = Structure_res(res,ci,src_num,pvs_num,obs_num)
    print Synthesis_performance(gt,parameter)
    #Draw_figure(parameter,para_ci)
    #Outlier_analysis(parameter,para_ci)

def Run_real_experiment():
    path = "./HRC_Model/database/"
    filename = "3DTV"
    data = ReadFile(path+filename+'.npy')
    
    (src_num, hrc_num, obs_num) = data.shape
    pvs_num = src_num*hrc_num
    #    # optimal method includes: Nelder-Mead, BFGS, Newton-CG, SLSQP
    #opt_method = 'trust-constr'
    opt_method = 'SLSQP'
    res,ci = MLE_process(data,src_num,hrc_num,pvs_num,obs_num,opt_method)
    parameter,para_ci = Structure_res(res,ci,src_num,pvs_num,obs_num)
    Draw_figure(parameter,para_ci)
    Outlier_analysis(parameter,para_ci)
    #hess_m = Hessian_L(res.x)
    #np.savez(opt_method+"_Jsureal_"+filename+'.npz', parameter, para_ci, hess_m)

def Run_MLE_MOS_CI_test(filename1,filename2):
    
 
    src_num = 9
    hrc_num = 16
    obs_num = 24
    pvs_num = src_num*hrc_num
    xe,bs,vs,ae,data = Synthesis_data(src_num,hrc_num,pvs_num,obs_num)
    
    gt = (xe,bs,vs,ae)

#    # optimal method includes: Nelder-Mead, BFGS, Newton-CG, SLSQP
    #opt_method = 'trust-constr'
    opt_method = 'SLSQP'
    res,ci = MLE_process(data,src_num,hrc_num,pvs_num,obs_num,opt_method)
    parameter,para_ci = Structure_res(res,ci,src_num,pvs_num,obs_num)
    
    ci_xe_correct = CI_performance_MLE(gt,parameter,para_ci,'xe')
    ci_xe_correct = ci_xe_correct.T
#    #return ci_xe_correct
    Write_file_csv(ci_xe_correct,filename1)
    
    ci_mos_correct = CI_performance_MOS(gt,data)
    ci_mos_correct = ci_mos_correct.T
    Write_file_csv(ci_mos_correct,filename2)

def Draw_MLE_MOS_CI_results(filename1, filename2):
    mle_ci_pvs, mle_ci_iteration, mle_ci_all, mos_ci_pvs,mos_ci_iteration,mos_ci_all = Analysis_Compare_CI_performance(filename1,filename2)
    Draw_res_CI_performance(mos_ci_iteration,'mean_ite','MOS')
    Draw_res_CI_performance(mle_ci_iteration,'mean_ite','MLE')
    Draw_res_CI_performance(mos_ci_pvs,'mean_pvs','MOS')
    Draw_res_CI_performance(mle_ci_pvs,'mean_pvs','MLE')
    print np.mean(mle_ci_pvs), np.mean(mle_ci_iteration),np.mean(mos_ci_pvs),np.mean(mos_ci_iteration)
   
def MLE_Boosting_CI_test():
    global src_num
    global hrc_num
    global obs_num
    global pvs_num
    global data
    
    src_num = 24
    hrc_num = 8
    obs_num = 28
    pvs_num = src_num*hrc_num
    xe,bs,vs,ae,data = Synthesis_data(src_num,hrc_num,pvs_num,obs_num)
    gt = (xe,bs,vs,ae)
    boot_res = Bootstrapping_analysis(gt,data)
    print np.mean(boot_res)

def Simulate_MOS_CI_only():
    global src_num
    global hrc_num
    global obs_num
    global pvs_num
    src_num = 24
    hrc_num = 8
    obs_num = 28
    pvs_num = src_num*hrc_num
    res = np.zeros((100,pvs_num))
    for i in range(100):
        xe,bs,vs,ae,data = Synthesis_data(src_num,hrc_num,pvs_num,obs_num)   
        gt = (xe,bs,vs,ae)    
        res[i,:] = CI_performance_MOS(gt,data)
    np.savetxt('simulate_mos_ci_only.csv', res, delimiter = ',')
    Draw_MLE_MOS_CI_results('simulate_mle_ci_only.csv', 'simulate_mos_ci_only.csv')

def Simulate_MLE_CI_only():
    global src_num
    global hrc_num
    global obs_num
    global pvs_num
    global data
    src_num = 24
    hrc_num = 8
    obs_num = 28
    pvs_num = src_num*hrc_num
    
    for i in range(100):
        xe,bs,vs,ae,data = Synthesis_data(src_num,hrc_num,pvs_num,obs_num)   
        gt = (xe,bs,vs,ae)
        res,ci = MLE_process(data,src_num,hrc_num,pvs_num,obs_num,'SLSQP')
        parameter,para_ci = Structure_res(res,ci,src_num,pvs_num,obs_num)
    
        ci_xe_correct = CI_performance_MLE(gt,parameter,para_ci,'xe')
        ci_xe_correct = ci_xe_correct.T
        
        Write_file_csv(ci_xe_correct,'simulate_jing_mle_ci_only.csv')
    Draw_MLE_MOS_CI_results('simulate_jing_mle_ci_only.csv', 'simulate_mle_ci_only.csv')    

def main():

    #### run synthesis experiment
#    print 'Newton-CG\n'
#    Run_synthesis_experiment('Newton-CG')
#    print "SLSQP\n"
#    Run_synthesis_experiment('SLSQP')
#    print "trust-constr\n"
#    Run_synthesis_experiment('trust-constr')
    
    
    #### run real experiment data
    #Run_real_experiment()
    
    #### run CI test on MOS and MLE
#    Simulate_MOS_CI_only()
#    Simulate_MLE_CI_only()
    
    #### run MLE Bootstrapping CI test
    MLE_Boosting_CI_test()
    
if __name__ == '__main__':
  
    main()
    print 'Done.'   


