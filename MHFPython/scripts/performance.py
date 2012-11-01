# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:41:39 2012

@author: eba
"""

from MHFPython import *
from time import *

original = GaussianHypothesis3();
assign(original.mean,[0,0,0]);
assign(original.cov,[0.2,0,0, 0,0.2,0, 0,0,1]);
original.weight = 1;

t = odometryTransformation(0.23,0.01,0)
m = beaconMeasurement(Matrix21([[1],[1]]))

limit = 2
steps = 200

inp = Matrix21([[1],[1]])
noisecov = Matrix22([[1,0],[0,1]])

begin = time()
samples = drawSamples(original,1000000)
print time()-begin
propagateSamples(samples,t.trans(),inp,noisecov)
print time()-begin
z = layoutSamples(samples,limit,steps)
print time()-begin

meas = Matrix11([[1]]);
measnoisecov = Matrix11([[0.01]])

updateSamples(samples,m.trans(),meas,measnoisecov)
print time()-begin
z = layoutSamples(samples,limit,steps)
print time()-begin
