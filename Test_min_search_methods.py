#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:54:08 2018

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




""" generate simulated data """
global pvs_num
global obs_num
global src_num

"""Zhi's MLE synthetic data setup
synthetic_result={
            'quality_scores': np.random.uniform(1, 5, 79),
            'observer_bias': np.random.normal(0, 1, 30),
            'observer_inconsistency': np.abs(np.random.uniform(0.0, 0.4, 30)),
            'content_bias': np.random.normal(0, 0.00001, 9),
            'content_ambiguity': np.abs(np.random.uniform(0.4, 0.6, 9)),
        }

"""


""" simulation parameter setup """
src_num = 24
hrc_num = 8
pvs_num = src_num*hrc_num
obs_num = 28

data = np.zeros((src_num,hrc_num,obs_num))
## do not change this part
""" true score"""
xe = np.random.uniform(1,5,pvs_num)

""" observer bias """
bs = np.random.uniform(-2,2,obs_num)
#bs = np.random.normal(0, 1, obs_num)

""" observer inconsistency """
vs = np.random.uniform(0.1,1,obs_num)


""" content ambicuity """
ae = np.random.uniform(0.1,1,src_num)


""" synthesized observed score """
global xes
xes =[[0 for i in range(pvs_num)]for i in range(obs_num)]
xes = np.array(xes)

pvs = -1
src_idx = np.zeros(pvs_num)
for src in range(0,src_num):
    for hrc in range(0,hrc_num):
        pvs = pvs+1
        src_idx[pvs] = src
               
np.random.seed(0)
pvs = -1
for src in range(0,src_num):
    for hrc in range(0,hrc_num):
        pvs = pvs+1
        for obs in range(0,obs_num):
            mu = xe[pvs]+bs[obs]
            sigma = np.sqrt(np.power(vs[obs],2)+np.power(ae[src],2))
            xes[obs,pvs] = np.random.normal(mu, sigma, 1)
            data[src,hrc,obs] = xes[obs,pvs]
# finish the synthesis process

# calculate src mos
srcmos = [[0 for i in range(hrc_num*obs_num)]for i in range(src_num)]
srcmos = np.array(srcmos)
pvs = -1
for src in range(0,src_num):
    n = -1
    for hrc in range(0,hrc_num):
        pvs = pvs+1
        for obs in range(0,obs_num):
            n=n+1;
            srcmos[src,n]=xes[obs,pvs]
            

""" likelihood function and its first-order derivatives"""

def Lfunction(x):
    """ the log likelihood function for MLE subjective model"""
    
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    
    res = 0
    pvs = -1
    
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            pvs = pvs+1
            for obs in range(0,obs_num):
                res = res + np.log(vs[obs]**2+ae[src]**2)+np.power((xes[obs,pvs]-xe[pvs]-bs[obs]),2)/(vs[obs]**2+ae[src]**2)
    return res

def Lfunction_det(x):
    """ the log likelihood function considering hessian matrix semi-definite for MLE subjective model"""
    
    xe = x[range(0,pvs_num)]
    bs = x[range(pvs_num,pvs_num+obs_num)]
    vs = x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
    ae = x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)]
    u = x[-1]
    
    
    res = 0
    pvs = -1
    
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            pvs = pvs+1
            for obs in range(0,obs_num):
                res = res + np.log(vs[obs]**2+ae[src]**2)+np.power((xes[obs,pvs]-xe[pvs]-bs[obs]),2)/(vs[obs]**2+ae[src]**2)
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
            
    
    e=-1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            init = 0
            e=e+1
            for s in range(0,obs_num):
                init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
            dxe[e] = init
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
        dbs[s] = init  
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init + 2*vs[s]/(vs[s]**2+ae[src]**2) -2*vs[s]*np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        dvs[s] = init  
            
    for src in range(0,src_num):
        init = 0
        for hrc in range(0,hrc_num):
            for s in range(0,obs_num):                 
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
            
    
    e=-1
    for src in range(0,src_num):
        for hrc in range(0,hrc_num):
            init = 0
            e=e+1
            for s in range(0,obs_num):
                init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
            dxe[e] = init
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init -2*(data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2)
        dbs[s] = init  
           
    for s in range(0,obs_num):
        init = 0
        for src in range(0,src_num):
            for hrc in range(0,hrc_num):
                init = init + 1/(vs[s]**2+ae[src]**2) -np.power((data[src,hrc,s]-xx[src,hrc]-bs[s])/(vs[s]**2+ae[src]**2),2)
        dvs[s] = init  
            
    for src in range(0,src_num):
        init = 0
        for hrc in range(0,hrc_num):
            for s in range(0,obs_num):                 
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
    pvs = -1
    for hrc in range(hrc_num):
        for src in range(src_num):
            init = 0
            pvs = pvs+1
            for obs in range(obs_num):
                init =init + 2/(vs[obs]**2+ae[src]**2)
                dxebs[pvs,obs] = 2/(vs[obs]**2+ae[src]**2)
                dxevs[pvs,obs] = 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
            dxe2[pvs,pvs] = init
    pvs = -1            
    for hrc in range(hrc_num):
        for src in range(src_num):
            init = 0
            pvs = pvs+1
            for obs in range(obs_num):
                init = init + 2*(data[src,hrc,obs]-xx[src,hrc]-bs[obs])/((vs[obs]**2+ae[src]**2)**2)
            dxeae[pvs,src] = init
    ## bs
    dbsxe = dxebs.T
    for obs in range(obs_num):
        dbs2[obs,obs] = 2*np.sum(1/(vs[obs]**2+ae**2))
        init = 0
        
        for src in range(src_num):
            init2 = 0
            for hrc in range(hrc_num):
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

nonlinear_constraint = NonlinearConstraint(cons_f, 0, np.inf, jac='2-point', hess=BFGS())
   
""" Test the proposed subject model """   
    
""" Initialization value for xe,bs,vs, ae """
xe0 = np.mean(xes,0) # calculate column mean
bs0 = np.zeros(obs_num)
vs0 = np.std(xes,axis=1,ddof=1)
ae0 = np.std(srcmos,axis=1,ddof=1)

xinput = np.r_[xe0,bs0,vs0,ae0]

""" set lower and upper bounds for each parameter """
tempstd = np.std(srcmos,axis=1,ddof=1)
srcusestd = np.max(tempstd)

tempstd = np.std(xes,axis=1,ddof=1)
obsusestd = np.max(tempstd)

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


"""using minimize method to find MLE estimates """
## nelder-mead method is worse than BFGS
#res = minimize(Lfunction, xinput, method='nelder-mead',options={'xtol':1e-8, 'disp':True})
#res = minimize(Lfunction, xinput, method='BFGS',jac = L_dev, options={'disp':True})
#res = minimize(Lfunction, xinput, method='Newton-CG',jac = L_dev2,options={'disp':True})
#res = minimize(Lfunction, xinput, method='Newton-CG',jac = L_dev, hess = Hessian_L, options={'disp':True})

""" with constraint that vs and ae >0 """
""" This method is significantly better than the method without constraints """
res = minimize(Lfunction, xinput, method='trust-constr',  jac=L_dev, hess = Hessian_L, constraints=[nonlinear_constraint], options={'verbose': 1}, bounds=bounds)

#res = minimize(Lfunction, xinput, method='SLSQP', jac = '2-point', options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
#res = minimize(Lfunction, xinput, method='SLSQP', jac = L_dev, options={'ftol': 1e-9, 'disp': True}, bounds=bounds)

"""consider using constraints hessian matrix semi-definite """
#res = minimize(Lfunction_det, np.r_[xinput,0.001], method='nelder-mead',options={'xtol':1e-8, 'disp':True})

print res
############# end of the optimization process##############

xee = res.x[range(0,pvs_num)]
bse = res.x[range(pvs_num,pvs_num+obs_num)]
vse = res.x[range(pvs_num+obs_num,pvs_num+obs_num+obs_num)]
aee = res.x[range(pvs_num+2*obs_num,pvs_num+2*obs_num+src_num)] 

plcc_observe = pearsonr(xe,xe0)
plcc_estimate = pearsonr(xe,xee)
plcc_obs_est = pearsonr(xe0,xee) 

rmse_observe = np.sqrt(np.mean((xe-xe0)**2)) 
rmse_estimate = np.sqrt(np.mean((xe-xee)**2))
print 'plcc between observed score xes and the gt score xe is',plcc_observe[0]
print 'plcc between estimated score xee and the gt score xe is',plcc_estimate[0]
print 'RMSE between observed score xes and the gt score xe is',rmse_observe
print 'RMSE between estimated score xee and the gt score xe is',rmse_estimate

rmse_bias = np.sqrt(np.mean((bs-bse)**2)) 
rmse_inconsistency = np.sqrt(np.mean((vs-vse)**2))

rmse_content = np.sqrt(np.mean((ae-aee)**2)) 
print 'RMSE of bias', rmse_bias
print 'RMSE of inconsistency', rmse_inconsistency
print 'RMSE of content ambiguity', rmse_content

plt.scatter(xe,xe0,marker=r'$\clubsuit$')
plt.xlabel("ground truth score xe ")
plt.ylabel("observed score, xes")
plt.show()

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