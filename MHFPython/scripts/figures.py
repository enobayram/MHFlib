# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:30:49 2012

@author: eba
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *
from MHFPython import *
from plotGaussians import *

def rotate(x,y,rot):
    return [x*rot[0,0]+y*rot[0,1],x*rot[1,0]+y*rot[1,1]]
    

theta = arange(-pi,pi,0.01)
r = 1;
limit = 3.;

[x,y] = [r*cos(theta), r*sin(theta)]

rotOrig = array([[1.,0.],[1.,1.]])
[xorig,yorig] = rotate(x,y,rotOrig)

variances = [0.025,1]
stddevs = sqrt(variances)

rotMax = array([[stddevs[0],0],[0,stddevs[1]]])
[xmax,ymax] = rotate(x,y,rotMax)


figure(1)
hold(False)
Orig = plot(xorig,yorig)
hold(True)
Max = plot(xmax,ymax)
ylim([-limit,limit])
xlim([-limit,limit])
grid(True)

legend([Orig,Max],["Original Gaussian","Maximum Component Size"])
title("2D Gaussian to Split")

rotOrigScaled = inv(rotMax).dot(rotOrig)
[xorigs,yorigs] = rotate(x,y,rotOrigScaled)

rotMaxScaled = inv(rotMax).dot(rotMax)
[xmaxs,ymaxs] = rotate(x,y,rotMaxScaled)


figure(2)
hold(False)
OrigS = plot(xorigs,yorigs)
hold(True)
MaxS = plot(xmaxs,ymaxs)
ylim([-limit,limit])
xlim([-limit,limit])
grid(True)

legend([OrigS,MaxS],["Original Gaussian","Maximum Component Size"])
title("Scaled Coordinates")

POrigScaled = rotOrigScaled.dot(rotOrigScaled.transpose());
eigs,rotDecompose = eig(POrigScaled)

rotIndependent = inv(rotDecompose).dot(rotOrigScaled)
[xind,yind] = rotate(x,y,rotIndependent)

figure(3)
hold(False)
OrigI = plot(xind,yind)
hold(True)
MaxI = plot(xmaxs,ymaxs)
ylim([-limit,limit])
xlim([-limit,limit])
grid(True)

legend([OrigI,MaxI],["Original Gaussian","Maximum Component Size"])

table = SplitTable1('tables/kl1e-1table');

plotvar = eigs[0];
#1D plot of the table entries
lim1d = sqrt(plotvar)*4;
x = arange(-lim1d,lim1d,lim1d/500)
#y = 1/sqrt(2*pi*originalVar)*exp(-1/2*(x*x)/originalVar)
y = 1/sqrt(2*pi*plotvar)*exp(-x*x/(2*plotvar))
fig=figure(4)
hold(False)
orig1d = plot(x, y)
hold(True)

y = zeros_like(x)
entry = table.getUpperEntry(plotvar)
entryvar = entry.variance;
varRatio = plotvar/entryvar;
hyp1d = entry.hypotheses
for gh in hyp1d:
    var = gh.cov(0)*varRatio;
    mean = gh.mean(0)*sqrt(varRatio);
    y=1/sqrt(2*pi*var)*exp(-(x-mean)*(x-mean)/(2*var))*gh.weight
    components = plot(x, y, color = 'green')
#savefig('figures/split')

legend([OrigI,MaxI],["Original","Components"])


vO = rotOrig.dot(rotOrig.transpose())

original = GaussianHypothesis3();
assign(original.mean,[0,0,0]);
assign(original.cov,[vO[0,0],vO[0,1],0, vO[1,0],vO[1,1],0, 0,0,1]);
original.weight = 1;

variancesMat = MeanMatrix();
assign(variancesMat, [variances[0],variances[1],2]);

result = GH3list();

mhf = MultiHypDist3();

split(original, result, variancesMat, table);

[x,y] = [r*cos(theta), r*sin(theta)]

figure(5)
hold(False)
Orig = plot(xorig,yorig)
hold(True)
for gh in result:
    mean = pyArray(gh.mean)
    rotGh = cholesky(pyArray(gh.cov))
    [xgh,ygh] = rotate(x,y,rotGh[0:2,0:2])
    [xghm, yghm] = [xgh+mean[0], ygh+mean[1]]
    plot(xghm,yghm, color='green')
ylim([-limit,limit])
xlim([-limit,limit])
legend([OrigI,MaxI],["Original","Components"])
grid(True)

steps = 100


plotGaussian(original,[0,1],limit,steps,6)
plotGaussians(result,[0,1], limit, steps, 7, 0.)
