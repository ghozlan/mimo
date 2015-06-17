# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:24:38 2015

Multi-Input Multi-Output (MIMO) Channels
Single User

@author: Hassan
"""

from __future__ import division #makes float division default. for integer division, use //. e.g. 7//4=1

import matplotlib
matplotlib.use('Agg') # force matplotlib to not use any Xwindows backend.
# fixes the error "no display name and no $DISPLAY environment variable"
# when code is run on Amazon Elastic Cloud Compute (EC2)

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

def rayleigh(m,n):
    return sqrt(1/2) * (standard_normal((m,n)) + 1j * standard_normal((m,n)))

#%%

# r = H x + z
# y = G r

def info_rate_opt(G,H,Sigma_Z):
    R = \
    log(det( 
    G.dot( H.dot(H.conj().transpose()) + Sigma_Z ).dot( G.conj().transpose() ) 
    )) \
    -  \
    log(det( 
    G.dot( Sigma_Z ).dot( G.conj().transpose() ) 
    ))
    
    return real(R)


def info_rate(G_RX,H,Sigma_Z):
    m, n = H.shape
    R_total = array(0)
    for i in range(n):
        G = G_RX[i,:].reshape((1,m))

        R_xx = identity(n)
        Sigma_Y = G.dot( H.dot(R_xx).dot(H.conj().transpose()) + Sigma_Z ).dot( G.conj().transpose() ) 
        
        R_xx_tilde = R_xx
        R_xx_tilde[i,i] = 0
        Sigma_Y_X = G.dot( H.dot(R_xx_tilde).dot(H.conj().transpose()) + Sigma_Z ).dot( G.conj().transpose() ) 

        H_Y = sum(log(eigvalsh(Sigma_Y)))
        H_Y_X = sum(log(eigvalsh(Sigma_Y_X)))
        R = H_Y - H_Y_X    

        R_total = R_total + R
    return real(R_total)
    
#%%

def rx_matrix(H,RX):
    if RX == 'MF':
        G = H.conj().transpose()
    elif RX == 'ZF':
        G = inv(H)  
    elif RX == 'MMSE':
        G = (H.conj().transpose()).dot(
        H.dot(H.conj().transpose()) + 
        Sigma_Z)
    return G

if __name__ == '__main__':
    NN = 2    
    H = rayleigh(NN,NN)
    SNR_dB = arange(-2,10,0.5)
    
    R_ZEROS = zeros( len(SNR_dB) )
    #RX_LIST = ['OPT','MMSE','ZF','MF']
    RX_LIST = ['MMSE','ZF','MF']
    RX = dict()
    for rx in ['OPT']+RX_LIST: 
        RX[rx] = array(R_ZEROS)
    
    for snr_index in range(len(SNR_dB)):
        print "SNR = " + str(SNR_dB[snr_index]) + " dB"
        SNR = 10**(SNR_dB[snr_index]/10)    #signal to noise ratio (linear)        
        sigma2 = 1.0/SNR            #noise variance
    
        Sigma_Z = sigma2 * identity(NN)
    
        G = identity(NN)
        R = info_rate_opt(G,H,Sigma_Z)
        RX['OPT'][snr_index] = R
        print 'R = %.2f' %(R)
        
        for rx in RX_LIST:
            G_RX = rx_matrix(H, rx)
            R = info_rate(G_RX,H,Sigma_Z)        
            RX[rx][snr_index] = R
            print 'R = %.2f' %(R)
    
    # plot
    SNR = 10**(SNR_dB/10)
    R_SHANNON = log(1+SNR)
    line_style = {'OPT':'-','MMSE':'.-','ZF':'x-','MF':'+-'}
    for rx in ['OPT'] + RX_LIST:
        plot(SNR_dB,RX[rx],line_style[rx],label=rx)
    xlabel('SNR (dB)')
    ylabel('Rate (nats/symbol)')
    legend(loc='upper left')
