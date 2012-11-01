# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:20:10 2012

@author: eba
"""

from numpy import *
from scipy import *
from matplotlib.pyplot import *
from MHFPython import *

def stateAct(gh,trans,inp,noisecov,samples,limit, steps):
    z = zeros([steps,steps])
    cellArea = (2. * limit / steps) ** 2
    mean = pyArray(gh.mean)
    chol = linalg.cholesky(pyArray(gh.cov))
    for i in range(samples):
        sample = mean+ chol.dot(np.random.normal(0,1,[2,1]))
        noise = ScalarMatrix([[np.random.normal(0,pyArray(noisecov)[0][0])]])
        sample = pyArray(trans.statetrans(MeanMatrix(sample),noise,inp))
        if (limit>sample).all() and (sample>-limit).all():
            index = (sample * steps/2 / limit + steps/2).astype(int)
            z[index[0,0]][index[1,0]] += 1./(samples*cellArea);
    return z