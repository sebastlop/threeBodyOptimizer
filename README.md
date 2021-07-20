# threeBodyOptimize
This repo allow for calculations of two electron wavefunctions by optimizing the local energy. 

Please read: "Accurate and simple wavefunctions for the helium isoelectronic sequence with correct cusp conditions", K V Rodriguez, G Gasaneo and D M Mitnik.  J. Phys. B: At. Mol. Opt. Phys. 40 (2007) 3923–3939

The structure is the following:

-driver: provided for using simple

  -plain: pack of functions provided by D. Mitnik for calculation of local energy for the GR2 wave function
  
  -OOfunctions: implementation of a class and several attributes and methods for same calculations as in plain. Also iterator methods are provided and a loss function based on standard deviation of local energy for the sampling in the volume.
  
  -optimizer: a class for optimizing by a simple random search. Extension to other random based methods will be provided soon


OOfunctions:

class threeBodyGR2 : for instantiate the class, the volume, a NumPy array with shape(nPoints,3) must be provided. Each column correspond to coordinates r_1, r_2, r_12.

  some attributes: 
  - charge (fixed)
  - beta (probably variable) 
  - C200 (surely variable)      
  - wavefunction  ->the wavefunction at each point of the volume   
  - localEnergy   ->the local energy at each point of the volume
               
  
  some methods: 
  - loss(self) returns the std of the local energy for each pointin volume
  - recalculate(self) this perform the complete calculations of wavefunctions and local energy. Very useful when some attribute is modified
  - provideNewVolume(self, vol) for passing new points. Include a call to racalculate
  - calculateLocEnergy(self) Calculations local energy for all the points
              
class optimize:  This class receive a threeBodyGR2 object, and **kwargs

  some attributes: 
  - optimizable  -> list of attribute names of threeBodyGR2 object (provided in kwargs see below the way to pass this)
  - nDim         -> the dimension of the parameter space to optimize
  - history      -> dictionary with the parameters to be optimized and history of mimimum losses achieved on each epoch
                   
  methods:  
  - simpleRandomSerach(self, epochs=1, nSamples=100, center = [1,], radii = [1,]): -> nSamples is the number of parameters to be sampled, for two parameters there will be nSamples of pairs in each epoch. Each uniform distribution of samples will range from center-radii to center+radii. center and radii should be tuple or numpy array with nDim elements. The return consist in the last minimum for the loss achieved and the corresponding set of optimum parameters.
  
  
How to optimize: a driver is provided but some lines will be explained:

*first we create a threeBodyGR2 object in a volume (parameters are set by default as in bibliography):*

    wave = threeBodyGR2(volume)
   
*second we create a dictionary with parameters to variate. The key is not internally used, the values should be a list with the name of the parameter as it is named in the threeBodyGR2 class, in this case: 'coeff' and 'beta'. The other two elements are useless by now but provides the bounds where for variation of those parameters.*

    varParameters = {'C200':['coeff',0,3.5], 'beta':['beta',0,1]}
    
*finally we instantiate the optimizer passing the above created dictionary as kwargs*

    opt = optimize(wave, **varParameters )
    
 In order to perform the optimization we have to execute the method:
 
    res = opt.simpleRandomSerach(epochs=50, nSamples=100, center = [1,1], radii = [1,1]) 
    
  *clearly center and radii should have the same elements numbers as parameters are optimized* 
