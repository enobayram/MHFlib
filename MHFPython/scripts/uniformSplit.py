# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:24:31 2012

@author: eba
"""

from numpy import *
from scipy import *
from matplotlib.pyplot import *
from MHFPython import *
from plotGaussians import *

mean = Matrix31()
assign(mean,[0,0,0])
widths = Matrix31()
assign(widths,[20,20,1])

result = GH3list()

maxVars = Matrix31()
assign(maxVars,[1,1,1])

table = SplitTable1('tables/kl1e-1uniftable')

uniformSplit(mean,maxVars,result,maxVars,table)

plotGaussians(result,[0,1],1,100.0,0,0)