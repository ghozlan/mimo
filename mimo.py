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

#%%
