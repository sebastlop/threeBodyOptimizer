import numpy as np
import matplotlib.pyplot as plt
from time import time
from OOfunctions.functions import threeBodyGR2
from optimizer.optimizers import optimize, volumeSampler

# np.random.seed(15) #for reproducibility

nSamples = 1024

#slightly more inteligent volume samplig 
vs = volumeSampler(nSamples)
volume = vs.singleZDistribution()


wave = threeBodyGR2(volume)

varParameters = {'C200':['coeff',0,10], 'beta':['beta',0,10]}
opt = optimize(wave, **varParameters )

res = opt.variableStepRandomSerach(epochs=50, nSamples=512, center = [1,1], radii = [1,1])
print('minimum at: ', res[0], 'with loss value: ', res[1])

wave.coeff = res[0][0]
wave.beta = res[0][1]
wave.recalculate()
print(f'the local energy avg: {wave.localEnergy.mean()}')
print(f'the local Virial avg: {wave.potEnergy.mean()/wave.localKineticEnergy.mean()}')
print(f'Number of evaluations of the loss function: {wave.lossCounter}')

plt.plot(wave.localEnergy)
plt.show()