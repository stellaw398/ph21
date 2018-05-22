#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:59:18 2018

@author: apple
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#Part I#
def cointoss(H,n,*s): #n is number of flips, H probability of flips, r can set h
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
cnt2 = 0

for h in [0.3,0.45,0.55,0.6]:
    Ho = h 
    plt.figure(cnt2)
    cnt = 1
    for i in (2)**np.linspace(0,8,9):
        r = math.ceil(i*Ho)
        sigma = np.sqrt(Ho*(1-Ho)/i)
        Plist = []
        Glist = []
        G3list = []
        Hlist = []
        plt.subplot(3,3,cnt)
        for j in np.linspace(0,1,51):
            Hlist.append(j)
            prob = cointoss(j,i,r)
            Plist.append(prob)
            gauss = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-((j-Ho)**2)/(2*(sigma**2)))
            gauss3 = 1/(3*sigma*np.sqrt(2*np.pi))*np.exp(-((j-Ho)**2)/(2*((3*sigma)**2)))
            G3list.append(gauss3)
            Glist.append(gauss)
        Plist = Plist/sum(Plist) #normalize list of posterier probabilities
        Glist = Glist/sum(Glist) #normalize list of gaussian probabilities
        G3list = G3list/sum(G3list)
        plt.plot(Hlist,Plist,'b', label= 'uniform')
        plt.plot(Hlist,Glist,'r', label='gaussian')
        plt.plot(Hlist,G3list, 'g')
        plt.title('H = %.2f'%Ho)
        plt.xlabel('H')
        plt.ylabel('probability')
        cnt+= 1
    plt.tight_layout()
    plt.show()
    print 'the blue curve is the uniform prior, the red curve is the gaussian prior, the green curve is the 3 sigma gaussian'
    cnt2 +=1 
#Part II#
plt.plot(cnt2)
def flash(n):
    theta = np.pi*np.random.rand(int(n))-np.pi/2 # theta between +/- Pi/2
    x_k= np.tan(theta)+1.0 #x
    return x_k
cnt =1
A = np.linspace(-7,9,101)
b =1
for n in (2)**np.linspace(0,8,9):   
    x_i = flash(n)
    p = []
    for a in A:
        L = 0
        for i in x_i:
            L += np.log(b**2+(i-a)**2)
        prob = np.exp(-L)
        p.append(prob)
    p = p/sum(p) #normalize
    plt.subplot(3,3,cnt)
    plt.plot(A,p, 'g.')
    plt.xlabel('a (km)')
    plt.ylabel('probability')
    x_i_mean = np.mean(x_i)
    plt.title('n = %d, mean = %.1f'%(n,x_i_mean))
    cnt += 1
plt.tight_layout()
plt.show()
print 'the mean of {x_k} is not a good estimator because the thetas are randomly distributed'
plt.plot(cnt2 + 1)
n = 128
x_n = flash(n)
B = np.linspace(4,0, 100, endpoint = False)
p = np.empty((len(B),len(A)))
for ib in range(len(B)):
    beta = B[ib]
    for ia in range(len(A)):
        alpha = A[ia]
        L_ab = 0
        for x in x_n:
            L_ab += np.log(beta**2+((x-alpha)**2))
        p[ib,ia] =  n* np.log(beta) - L_ab
p = p-np.amax(p)
p =np.exp(p)
p = np.exp(p)
plt.contour(A,B,p)
plt.xlabel('a (km)')
plt.ylabel('b (km)')
plt.title('Posterior distribution in the a-b plane')
plt.colorbar()
plt.show()

    
            
