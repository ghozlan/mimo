# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:31:50 2015

Multi-Input Multi-Output (MIMO) Channels
Multi-User
Uplink

@author: Hassan
"""

from __future__ import division #makes float division default. for integer division, use //. e.g. 7//4=1

from numpy import *
from matplotlib.pyplot import *
from numpy.fft import ifft, fftshift, fft
from numpy.linalg import svd
from numpy.linalg import eigvalsh #eigvalsh : eigenvalues of a symmetric or Hermitian (conjugate symmetric)

from numpy.random import randint, standard_normal
from numpy.linalg import det
from numpy.linalg import inv # inverse
from numpy.linalg import pinv # psuedo-inverse

#%%

from mimo import rayleigh

#%%
# r = H1 x1 + H2 x2 + z
# y1 = G1 r
# y2 = G2 r

def rx_matrices(H1,H2,RX):
    G_RX = list()
    for H in [H1, H2]:
        if RX == 'MF':
            G = H.conj().transpose()
        elif RX == 'ZF':
            G = inv(H)  
        elif RX == 'MMSE':
            G = (H.conj().transpose()).dot(
            H1.dot(H1.conj().transpose()) + 
            H2.dot(H2.conj().transpose()) + 
            Sigma_Z)
        G_RX.append(G)
    return G_RX


def info_rate(G_RX,H,Sigma_Z): 
    #G_RX is a list for receicer processing matrices for all users
    #H is a list for channel matrices
    H1, H2 = H
    
    R_users = list()
    for u in [0,1]: # 2 users
        m, n = H[u].shape
        R_total = array(0)
        for i in range(n):
            G = G_RX[u][i,:].reshape((1,m))
    
            R_x1 = identity(n)
            R_x2 = identity(n)
            Sigma_Y = G.dot( 
            H1.dot(R_x1).dot(H1.conj().transpose()) + 
            H2.dot(R_x2).dot(H2.conj().transpose()) + 
            Sigma_Z 
            ).dot( G.conj().transpose() ) 
            
            if u==1: # if user 1
                R_x1[i,i] = 0
            else: # if user 2
                R_x2[i,i] = 0
    
            Sigma_Y_X = G.dot( 
            H1.dot(R_x1).dot(H1.conj().transpose()) + 
            H2.dot(R_x2).dot(H2.conj().transpose()) + 
            Sigma_Z 
            ).dot( G.conj().transpose() ) 
            
            H_Y = sum(log(eigvalsh(Sigma_Y)))
            H_Y_X = sum(log(eigvalsh(Sigma_Y_X)))
            R = H_Y - H_Y_X    
    
            R_total = R_total + R
            
        R_users.append(R_total)          
    return R_users

#%%
    
NN = 2

H1 = rayleigh(NN,NN)
H2 = rayleigh(NN,NN)

SNR_dB = arange(-2,10,0.5) # -2 to 4, increment = 0.5
R_ZEROS = zeros( len(SNR_dB) )
#RX_LIST = ['OPT','MF','ZF','MMSE']
RX_LIST = ['MF','ZF','MMSE']
R_LIST = ['R1', 'R2', 'R_sum']
RX = dict()
for rx in RX_LIST: 
    RX[rx] = dict()
    for r in R_LIST: RX[rx][r] = array(R_ZEROS)

#%%
for snr_index in range(len(SNR_dB)):
    print "SNR = " + str(SNR_dB[snr_index]) + " dB"
    SNR = 10**(SNR_dB[snr_index]/10)    #signal to noise ratio (linear)        
    sigma2 = 1.0/SNR            #noise variance

    Sigma_Z = sigma2 * identity(NN)
    
    for rx in RX_LIST:
        G_RX = rx_matrices(H1, H2, rx)
        R1, R2 = info_rate(G_RX,[H1, H2],Sigma_Z)        
        #R1, R2 = 0, 0
        RX[rx]['R1'][snr_index] = R1
        RX[rx]['R2'][snr_index] = R2
        RX[rx]['R_sum'][snr_index] = R1 + R2
    
    #print 'R = %.2f' %(R)    

# plot
SNR = 10**(SNR_dB/10)
R_SHANNON = log(1+SNR)
#plot(SNR_dB,R_SHANNON,label='log(1+SNR)')
#plot(SNR_dB,R_OPT ,'-',label='OPT')
line_style = {'OPT':'-','MMSE':'.-','ZF':'x-','MF':'+-'}
for rx in RX_LIST:
    plot(SNR_dB,RX[rx]['R_sum'],line_style[rx],label=rx)
xlabel('SNR (dB)')
ylabel('Rate (nats/symbol)')
legend(loc='upper left')
