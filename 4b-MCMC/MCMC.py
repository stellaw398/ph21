#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:35:59 2018

@author: apple
"""

##Markov Chain Monte Carlo

##Part 1 Coin Flip

import math
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
def cointoss(H,n,*s): #n is number of flips, H probability of flips, s can set h
    toss = np.random.rand(int(n))
    h = 0
    if s != ():
        h = int(s[0])
    else:
        for i in toss:
            if i < H:
                h += 1
    prob = (math.factorial(n)/(math.factorial(h)*math.factorial(n-h)))*(H**h)*((1-H)**(n-h))
    return prob
H_true= 0.30
n = 2**7
r = math.ceil(n*H_true)
x = np.linspace(0,1,51)
y = cointoss(x,n,r)
def lnlike(theta,x,y):
    model = theta
    inv_sigma2 = n/(H_true*(1-H_true))
    return -0.5*(np.sum((H_true-model)**2*inv_sigma2-np.log(inv_sigma2)))
def lnprior(theta):
    b = theta
    if 0<b< 1.0:
        return 0.0
    return -np.inf
def lnprob(theta,x,y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x,y)
ndim=1 #, nwalkers = 1, 100
pos =  [1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x,y))
cnt=1
plt.figure(0)
for i in [10,50,100]:
    nwalkers = i
    for j in [100,250,500]:
        sampler.run_mcmc(pos,j)
        samples = sampler.chain[:,50:,:].reshape((-1,ndim))
        fig=corner.corner(samples,labels=['h'],truths= [H_true])
        plt.title('chain number = %d, chain length = %d'%(j,i))
        plt.show()

##Part II Lighthouse
def flash(n):
    theta = np.pi*np.random.rand(int(n))-np.pi/2 # theta between +/- Pi/2
    x_k= np.tan(theta)+1.0 #x
    return x_k

a_true = 1.0
b_true = 1.0
N = 2**8
X = flash(N)
B = np.linspace(4,0, 100, endpoint = False)
A = np.linspace(-7,9,101)
def lnlike(theta,x,y): 
    a,b =theta
    model= sum(np.log(b**2+((x-a)**2)))
    return N*np.log(b)-model 
def lnprior(theta):
    a,b = theta
    if 0<b<5.0:
        return 0.0
    return -np.inf
def lnprob(theta,x,y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,x,y)
ndim, nwalkers = 2,100
pos =  [1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(X,A))
sampler.run_mcmc(pos,500)
samples = sampler.chain[:,50:,:].reshape((-1,ndim))
fig1=corner.corner(samples, labels=['a','b'],truths = [a_true,b_true])
plt.show()

def interloper(n,a,b):
    theta = np.pi*np.random.rand(int(n))-np.pi/2
    x_k = b*np.tan(theta)+a
    return x_k
X = np.append(flash(N),interloper(2**9,4,3.984))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(X,A))
sampler.run_mcmc(pos,500)
samples = sampler.chain[:,50:,:].reshape((-1,ndim))
fig2=corner.corner(samples,labels=['a','b'], truths = [a_true,b_true])
plt.show()
