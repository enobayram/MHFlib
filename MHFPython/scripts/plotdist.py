# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:56:15 2012

@author: eba
"""

from MHFPython import *
from plotGaussians import *
from plotLines import *

mhd = MultiHypDist3();
mhd.read('dists/dist3');

plotGaussians(mhd.GHlist,[0,1], 3.,80.,5, 0)
plotLines();