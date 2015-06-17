# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:31:50 2015

Multi-Input Multi-Output (MIMO) Channels
Multi-User
Downlink
2-antenna TX and 2 2-antenna RXs

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
# r1 = H1 x + z1
# r2 = H2 x + z2
# x = x1 + x2
# x1 = G1 v1
# x2 = G2 v2

def precoding_matrices(H1,H2,RX):
    G_RX = list()
    for H in [H1, H2]:
        if RX == 'MF':
            G = H.conj().transpose()
        elif RX == 'ZF':
            G = inv(H)  
        elif RX == 'MMSE':
            #R_xx = identity(2) #?XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            G = (H.conj().transpose()).dot(inv(
            H.dot(H.conj().transpose()) + 
            Sigma_Z))
        G_RX.append(G)
    return G_RX

        
def info_rate(G_PRE,H,Sigma_Z): 
    #G_PRE is a list for precoding matrices for all users
    #H is a list for channel matrices
    G1, G2 = G_PRE
    
    R_users = list()
    for u in [0,1]: # 2 users
        m, n = H[u].shape
        R_total = array(0)
        for i in range(n):
            #G = G_RX[u]
            
            #R_v1 = identity(n)
            #R_v2 = identity(n)
            R_v1 = zeros((n,n))
            R_v1[0,0] = 1
            R_v2 = zeros((n,n))
            R_v2[1,1] = 1

            Sigma_X = \
            G1.dot(R_v1).dot(G1.conj().transpose()) + \
            G2.dot(R_v2).dot(G2.conj().transpose())
            
            Sigma_Y = ( 
            H[u].dot(Sigma_X).dot(H[u].conj().transpose()) + 
            Sigma_Z 
            )
            
            if   u==0: # if user 1
                R_v1[i,i] = 0
            elif u==1: # if user 2
                R_v2[i,i] = 0

            Sigma_X_V = \
            G1.dot(R_v1).dot(G1.conj().transpose()) + \
            G2.dot(R_v2).dot(G2.conj().transpose())
            
            Sigma_Y_V = ( 
            H[u].dot(Sigma_X_V).dot(H[u].conj().transpose()) + 
            Sigma_Z 
            )
            
            H_Y = sum(log(eigvalsh(Sigma_Y)))
            H_Y_V = sum(log(eigvalsh(Sigma_Y_V)))
            R = H_Y - H_Y_V    
    
            R_total = R_total + R
            
        R_users.append(R_total)          
    return R_users

#%%
def info_rate_winterference(G_PRE,H,Sigma_Z): 
    #G_PRE is a list for precoding matrices for all users
    #H is a list for channel matrices
    G1, G2 = G_PRE
    
    R_users = list()
    for u in [0,1]: # 2 users
        m, n = H[u].shape
        R_total = array(0)
        for i in range(n):
            #G = G_RX[u]
            
            R_v1 = identity(n)
            R_v2 = identity(n)
            
            Sigma_X = \
            G1.dot(R_v1).dot(G1.conj().transpose()) + \
            G2.dot(R_v2).dot(G2.conj().transpose())
            
            Sigma_Y = ( 
            H[u].dot(Sigma_X).dot(H[u].conj().transpose()) + 
            Sigma_Z 
            )
            
            if   u==0: # if user 1
                R_v1[i,i] = 0
            elif u==1: # if user 2
                R_v2[i,i] = 0

            Sigma_X_V = \
            G1.dot(R_v1).dot(G1.conj().transpose()) + \
            G2.dot(R_v2).dot(G2.conj().transpose())
            
            Sigma_Y_V = ( 
            H[u].dot(Sigma_X_V).dot(H[u].conj().transpose()) + 
            Sigma_Z 
            )
            
            H_Y = sum(log(eigvalsh(Sigma_Y)))
            H_Y_V = sum(log(eigvalsh(Sigma_Y_V)))
            R = H_Y - H_Y_V    
    
            R_total = R_total + R
            
        R_users.append(R_total)          
    return R_users

#%%
    
NN = 2

H1 = rayleigh(NN,NN)
H2 = rayleigh(NN,NN)

SNR_dB = arange(-2,10,0.5) # -2 to 4, increment = 0.5
R_ZEROS = zeros( len(SNR_dB) )
RX_LIST = ['MMSE','ZF','MF']
R_LIST = ['R1', 'R2', 'R_sum']
R_LIST = ['R1', 'R2', 'R_sum', 'R_sum2']
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
        G_PRE = precoding_matrices(H1, H2, rx)
        R1, R2 = info_rate(G_PRE,[H1, H2],Sigma_Z)        
        #R1, R2 = 0, 0
        RX[rx]['R1'][snr_index] = R1
        RX[rx]['R2'][snr_index] = R2
        RX[rx]['R_sum'][snr_index] = R1 + R2
        RX[rx]['R_sum2'][snr_index] = sum(info_rate_winterference(G_PRE,[H1, H2],Sigma_Z))
    
    #print 'R = %.2f' %(R)    

# plot
SNR = 10**(SNR_dB/10)
R_SHANNON = log(1+SNR)
#plot(SNR_dB,R_SHANNON,label='log(1+SNR)')
#plot(SNR_dB,R_OPT ,'-',label='OPT')
line_style = {'OPT':'-','MMSE':'.-','ZF':'x-','MF':'+-'}
for rx in RX_LIST:
    plot(SNR_dB,RX[rx]['R_sum'],'b'+line_style[rx],label=rx)
    plot(SNR_dB,RX[rx]['R_sum2'],'r'+line_style[rx],label=rx)
xlabel('SNR (dB)')
ylabel('Rate (nats/symbol)')
legend(loc='upper left')
