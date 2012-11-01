# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:34:48 2012

@author: eba
"""

import MHFPython
reload(MHFPython)
from MHFPython import *
mhd = MultiHypDist2()
mhd.GHlist.push_front(GaussianHypothesis2())
class pythonimp(transformations):
     def statetrans(self, oldstate,noise,inp):
         print 'called again'
         return MeanMatrix([[1],[1]])

t = pythonimp()
inp = ScalarMatrix()
noisecov = ScalarMatrix()

UKFPredict(mhd,inp,noisecov,t)