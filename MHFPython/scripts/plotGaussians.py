# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:03:10 2012

@author: eba
"""

from numpy import *
from scipy import *
from matplotlib.pyplot import *
from MHFPython import *

def plotGaussian(gh,dims, limit, steps, fignum):
    dim0 = dims[0]
    dim1 = dims[1]
    x,y = ogrid[-limit:limit:2*limit/steps, -limit:limit:2*limit/steps]
    mean = pyArray(gh.mean)[ix_(dims)];
    cov = pyArray(gh.cov)[ix_(dims,dims)];
    
    icov = linalg.inv(cov);
    
    z = 1./(2.*pi*sqrt(linalg.det(cov))) \
        *exp(-1./2.*((x-mean[0])**2*icov[0,0] \
                    +2.*(x-mean[0])*(y-mean[1])*icov[0,1] \
                    +(y-mean[1])**2*icov[1,1]));
    maxa = max(z.flatten())
    #hold(True)
    # Creating image
    fig=figure(fignum)
    imshow(z.transpose(), origin='lower', extent=[-limit,limit,-limit,limit])
    fig.canvas.draw();
    fig.show()
    
    return maxa
    
def plotGaussians(ghlist,dims, limit, steps, fignum, maxa):
    x,y = ogrid[-limit:limit:2*limit/steps, -limit:limit:2*limit/steps]
    z=0*x+0*y;
    fig=figure(fignum)
    #counter=0;
    for hyp in ghlist:
        mean = pyArray(hyp.mean)[ix_(dims)];
        cov = pyArray(hyp.cov)[ix_(dims,dims)];
        icov = linalg.inv(cov);
        z += 1./(2.*pi*sqrt(linalg.det(cov))) \
             *exp(-1./2.*((x-mean[0])**2*icov[0,0] \
                         +2.*(x-mean[0])*(y-mean[1])*icov[0,1] \
                         +(y-mean[1])**2*icov[1,1]))*hyp.weight;
    z[0,0]=maxa;
    imshow(z.transpose(), origin='lower', extent=[-limit,limit,-limit,limit]);
        #counter += 1;
        #savefig('figures/fig%d'%counter)    
    fig.show();