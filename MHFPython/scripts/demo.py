# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/eba/.spyder2/.temp.py
"""
from numpy import *
from scipy import *
from matplotlib.pyplot import *
from MHFPython import *
from plotGaussians import *
from actualdist import *
import time
import os

originalVar = 0.02;
limit = 2.
steps = 100

original = GaussianHypothesis3();
assign(original.mean,[0,0,0]);
assign(original.cov,[originalVar,0,0, 0,originalVar,0, 0,0,1]);
original.weight = 1;

samples = drawSamples(original,1000000)

z = layoutSamples(samples,limit,steps)

fig=figure(1)
imshow(z.transpose(), origin='lower', extent=[-limit,limit,-limit,limit])
fig.canvas.draw();

maxa = z.max()
#maxa = plotGaussian(original,[0,1],limit,steps,1)

table = SplitTable1(); #use the default table, generated for a max kl divergence of 0.01

variances = MeanMatrix();
assign(variances, [0.1,0.1,0.02]);

result = GH3list();

mhf = MultiHypDist3();
singlehyp = MultiHypDist3();

split(original, result, variances, table);

plotGaussians(result,[0,1], limit, steps, 4, maxa)

if(not os.path.exists('./figures')):
    os.mkdir('figures')

savefig('figures/initial')


plotvar = 2;
#1D plot of the table entries
lim1d = sqrt(plotvar)*4;
x = arange(-lim1d,lim1d,lim1d/500)
#y = 1/sqrt(2*pi*originalVar)*exp(-1/2*(x*x)/originalVar)
y = 1/sqrt(2*pi*plotvar)*exp(-x*x/(2*plotvar))
fig=figure(3)
hold(False)
plot(x, y)
hold(True)

y = zeros_like(x)
entry = table.getUpperEntry(plotvar)
entryvar = entry.variance;
varRatio = plotvar/entryvar;
hyp1d = entry.hypotheses
for gh in hyp1d:
    var = gh.cov(0)*varRatio;
    mean = gh.mean(0)*sqrt(varRatio);
    y+=1/sqrt(2*pi*var)*exp(-(x-mean)*(x-mean)/(2*var))*gh.weight
plot(x, y)
savefig('figures/split')


# Plotting contour lines
#contour(z, origin='lower', extent=[-1,1,-1,1])

#xlabel('x')
#ylabel('y')
#title('A spiral !')

# Adding a line plot slicing the z matrix just for fun. 
#plot(x[:], z[50, :])


class pythonimp(transformations):
     def statetrans(self, oldstate,noise,inp):
         oldstate = pyArray(oldstate)
         newstate = oldstate*oldstate
         return MeanMatrix(newstate)

#t = pythonimp()
t = odometryTransformation(0.23,0.001,0)
inp = Matrix21([[1],[1]])
noisecov = Matrix22([[1,0],[0,1]])

mhf.GHlist.push_front(result)
singlehyp.GHlist.push_front(original)

UKFPredict(mhf,inp,noisecov,t)
UKFPredict(singlehyp,inp,noisecov,t)

plotGaussians(mhf.GHlist,[0,1], limit,steps,5, 0)
savefig('figures/mhfpredict')
plotGaussians(singlehyp.GHlist,[0,1], limit,steps,2, 0)
savefig('figures/ukfpredict')


propagateSamples(samples,t,inp,noisecov)
z = layoutSamples(samples,limit,steps)
maxb = z.max()
#z[0,0]=maxa
fig=figure(6)
imshow(z.transpose(), origin='lower', extent=[-limit,limit,-limit,limit])
fig.canvas.draw();
savefig('figures/actualpredict')


m = beaconMeasurement(Matrix21([[1],[1]]))

meas = Matrix11([[1]]);
measnoisecov = Matrix11([[0.01]])


UKFUpdate(mhf,meas,measnoisecov,m)
UKFUpdate(singlehyp,meas,measnoisecov,m)

plotGaussians(mhf.GHlist,[0,1], limit,steps,7, 0)
savefig('figures/mhfupdate')
plotGaussians(singlehyp.GHlist,[0,1], limit,steps,9, 0)
savefig('figures/ukfupdate')

updateSamples(samples,m,meas,measnoisecov)
z = layoutSamples(samples,limit,steps)

fig=figure(8)
imshow(z.transpose(), origin='lower', extent=[-limit,limit,-limit,limit])
fig.canvas.draw();
savefig('figures/actualupdate')
print "Figures generated in './figures/'"
raw_input("Press Enter to continue...")
