#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:45:31 2018

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt

def PCA(X):
    X_bar = np.mean(X,axis=1)
    X_prime = X.T - X_bar
    return np.linalg.eig(np.cov(X_prime.T))
a = 10.0
b = 2.0
y = np.linspace(0,10)
x = a + b*(y + np.random.rand(len(y)))
plt.figure(0)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')

X=np.stack((x,y))      
(PC,V)= PCA(X)
print V
print "The principal components are vectors that are parralel and perpendicular to the line"
plt.savefig('linear.pdf')
plt.close()

##Three camera problem
##generate data
##first place cameras along standard x,y,z axis
##SHM parameters
A = 5 # amplitude of oscillation
Xo = 10 ##equilibrium position
Yo = 5
Zo = 0
f = 120 #frequency
T = 10 #time in minutes
t = np.linspace(0,T*60,T*60*f,endpoint= 'False')##
w = 2*np.pi*f

f, (ax1,ax2,ax3) = plt.subplots(3)
##camera A- along y direction
CamA = (Xo-3,Yo+1,Zo)
X_a, Y_a, Z_a = CamA
XA = Xo - X_a + A*np.cos(w*t)
YA = np.ones(len(t))*(Y_a-Yo)
ZA = np.zeros(len(t))
ax1.scatter(XA,YA)
##camera B - along z direction
CamB = (Xo-4,Yo, Zo+4)
X_b, Y_b, Z_b = CamB
YB = Xo - X_b + A*np.cos(w*t)
XB = np.zeros(len(t))
ZB = np.ones(len(t))*Z_b
ax2.scatter(XB,YB)
##camera C - along x direction
CamC = (Xo+A+3, Yo, Zo)
X_c, Y_c, Z_c = CamC
XC = np.ones(len(t))*X_c
YC = np.ones(len(t))*Y_c
ZC = X_c - Xo - A*np.cos(w*t)
ax3.scatter(XC,YC)
f.tight_layout()
plt.show()

## Rotate cameras and add error
def Rx(gamma): #rotation matrix
    c, s = np.cos(gamma),np.sin(gamma)
    return np.array(((1,0,0),(0,c,-s),(0,s,c)))
def Ry(beta):     
    c, s = np.cos(beta),np.sin(beta)
    return np.array(((c,0,s),(0,1,0),(-s,0,c)))
def Rz(alpha):
    c, s = np.cos(alpha),np.sin(alpha)
    return np.array(((c,-s,0),(s,c,0),(0,0,1)))
def R(X,Y,Z,alpha,beta,gamma):
    Rt = np.matmul(Rz(alpha),Ry(beta),Rx(gamma))
    M = np.matmul(np.stack((X.T,Y.T,Z.T),axis=-1),Rt)
    return M.T
thetaA = np.array((np.radians(10),np.radians(4),np.radians(20)))
thetaB = np.array((np.radians(15),np.radians(11), np.radians(5)))
thetaC = np.array((np.radians(7),np.radians(19), np.radians(17)))
XA,YA,ZA = R(XA,YA,ZA,thetaA[0],thetaA[1],thetaA[2])
XB,YB,ZB = R(XB,YB,ZB,thetaB[0],thetaB[1],thetaB[2])
XC,YC,ZC = R(XC,YC,ZC,thetaC[0],thetaC[1],thetaC[2])
XA = XA + np.random.rand(len(t)) ##adding error
YA = YA + np.random.rand(len(t))
XB = XB + np.random.rand(len(t))
YB = YB + np.random.rand(len(t))
XC = XC + np.random.rand(len(t))
YC = YC + np.random.rand(len(t))
plt.figure(1)
plt.plot(XA,YA, 'r', XB, YB, 'b', XC,YC, 'g')
plt.title('Camera measurements')
plt.show()
X = np.stack((XA,YA,XB,YB,XC,YC))
(PC,V)= PCA(X)
Y = np.matmul(V.T,X)
plt.figure(2)
plt.plot(Y[0],Y[1],'r',Y[2],Y[3],'b',Y[4],Y[5],'g')
plt.xlabel('x')
plt.ylabel('y')
print 'this plot shows that only the x component varies'
plt.savefig('CameraProblem.pdf')
plt.close()

