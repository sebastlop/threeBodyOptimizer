import numpy as np
import matplotlib.pyplot as plt
from time import time
from OOfunctions.functions import threeBodyGR2
from optimizer.optimizers import optimize, volumeSampler

#np.random.seed(15) #for reproducibility

nSamples = 1000

#slightly more inteligent volume samplig 
vs = volumeSampler(nSamples)
volume = vs.singleZDistribution()


wave = threeBodyGR2(volume)

varParameters = {'C200':['coeff',0,3.5], 'beta':['beta',0,1]}
opt = optimize(wave, **varParameters )
res = opt.simpleRandomSerach(epochs=50, nSamples=100, center = [1,1], radii = [1,1])
print('minimum at: ', res[0], 'with loss value: ', res[1])

wave.coeff = res[0][0]
wave.beta = res[0][1]
wave.recalculate()
print(f'the local energy avg: {wave.localEnergy.mean()}')

plt.plot(wave.localEnergy)
plt.show()
