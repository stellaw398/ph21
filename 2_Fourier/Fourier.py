#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:07:52 2018

@author: apple
"""
import numpy as np
import matplotlib.pyplot as plt
##Intro to FFT
##Part I
C = 0
A = 5
f = 2*np.pi
phi = 0#-np.pi/2 
def cos(x):
    return C + A*np.cos(f*x+phi)
B=100
L= 0.5
def Gauss(x):
    return A*np.exp(-B*(x-L)**2)
t = np.arange(256.0)/256
fig0 = plt.figure(0)
spcos = np.fft.fft(cos(t))
invcos = np.fft.ifft(spcos)
freq1 = np.fft.fftfreq(t.shape[-1])
ax1 = fig0.add_subplot(311)
ax2 = fig0.add_subplot(312)
ax3 = fig0.add_subplot(313)
ax1.plot(t,cos(t))
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax2.plot(freq1,spcos)
ax2.set_xlabel('frequency')
ax2.set_ylabel('Power')
ax3.plot(t,invcos)
ax3.set_xlabel('time')
ax3.set_ylabel('Amplitude')
ax1.set_title("original cosine signal")
ax2.set_title("Cosine FFT")
ax3.set_title("recovered signal")
fig0.tight_layout()
plt.savefig('Cosine.pdf')
plt.close()

fig1 = plt.figure(1)
spgauss = np.fft.fft(Gauss(t))
invgauss = np.fft.ifft(spgauss)
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)
ax1.plot(t,Gauss(t))
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax2.plot(freq1,spgauss)
ax2.set_xlabel('frequency')
ax2.set_ylabel('Power')
ax3.plot(t,invgauss)
ax3.set_xlabel('time')
ax3.set_ylabel('Amplitude')
ax1.set_title("original Gaussian signal")
ax2.set_title("Gaussian FFT")
ax3.set_title("recovered signal")
fig1.tight_layout()
plt.savefig('Gauss.pdf')
plt.close()

##Part II
signal1 = np.loadtxt('arecibo1.txt')
samplet= np.arange(signal1.size)
fig2=plt.figure(2)
ax1 = fig2.add_subplot(311)
ax2 = fig2.add_subplot(312)
ax3 = fig2.add_subplot(313)
ax1.plot(samplet,signal1)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
sp= np.fft.fft(signal1)
freq = np.fft.fftfreq(samplet.shape[-1])
ax2.plot(freq,sp)
ax2.set_xlabel('frequency')
ax2.set_ylabel('Power')
ax3.plot(freq,sp)
ax3.set_xlim([.135,.14])
ax3.set_xlabel('frequency')
ax3.set_ylabel('Power')
ax1.set_title("Arecibo Signal")
ax2.set_title("FFT")
ax3.set_title("FFT near peak")
fig2.tight_layout()
plt.savefig('Arecibo.pdf')
plt.close()

x0=.13704
t0= 2000
def Gaussenv(x):
    return np.exp((-((x-x0)**2)/(dt**2)))
plt.figure(3)
for dt in [t0/4.0,t0/2.0, t0, 2*t0,4*t0]:
    spenv=np.fft.fft(Gaussenv(samplet))
    plt.plot(freq+0.13704,spenv,label = ("dt = %d"%dt))
plt.plot(freq, sp, label = 'Arecibo signal FFT') 
axes = plt.gca()
axes.set_xlim([.136,.138])
plt.xlabel('frequency')
plt.ylabel('Power')
plt.title("Comparison of Gaussian envelope dt")
plt.legend()
plt.tight_layout()
plt.savefig('envelope.pdf')
plt.close()

##Part III
from astropy.stats import LombScargle
plt.figure(4)
f0,p0 = LombScargle(t,Gauss(t)).autopower()
plt.plot(f0,p0)
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([-.25,1])
plt.xlabel('frequency')
plt.ylabel('Power')
plt.title('Gaussian Lombscargle')
plt.savefig('GaussLomb.pdf')
plt.close()


plt.figure(5)
samplet1 = np.linspace(0,signal1.size,signal1.size)
f1,p1 = LombScargle(samplet1,signal1).autopower()
plt.plot(f1,p1)
axes = plt.gca()
axes.set_xlim([.135,.14])
plt.xlabel('frequency')
plt.ylabel('Power')
plt.title('Arecibo Lombscargle')
plt.savefig('AreciboLomb.pdf')
plt.close()

plt.figure(6)
herx1 = np.loadtxt('herx1.txt')
hert = [row[0] for row in herx1]
hermag = [row[1] for row in herx1]
f2,p2 = LombScargle(hert,hermag).autopower(minimum_frequency=0.01, maximum_frequency =1)
plt.plot(f2,p2)
plt.xlabel('frequency')
plt.ylabel('Power')
plt.title('HerX1 Lombscargle')
plt.savefig('HerLomb.pdf')
plt.close()