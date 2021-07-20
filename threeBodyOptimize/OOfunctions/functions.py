# Library for computing functions and methods to calculate Local Hamilton Operator
# Uses Numpy 
# @Sebastián D. López version 
# 15/7/2021

import numpy as np

class threeBodyGR2:
    '''This class compute the 3 body wavefunction GR2 and the local energy defined for a particular point 
        or a set of points in an hyper volume. 
        Parameters are: - charge (fixed)
                        - beta (probably variable)
                        - C200 (surely variable)
        variables: r1, r2, r12. Provided in a 2D NumPy array [[r1 r2 r12] [r1 r2 r12] ....]
       
        Attributes: - Wave related:
                        - norma:        the wave norm
                        - fi0:          not normalized hydrogenic 1s in r1
                        - fi1:          not normalized hydrogenic 1s in r2
                        - xi:           distoriom in r12
                        - omega:        correlation in r1, r2, r12
                        - phi:          fi0 * fi1 * xi
                        - wavefunction  norma * phi * omega
                    
                    - First derivatives related:
                        - DiOmegaOnAxis0:   derivative of omega in r1
                        - DiOmegaOnAxis1:   derivative of omega in r2
                        - DiOnAxis0:        derivative of wavefunction in r1
                        - DiOnAxis1:        derivative of wavefunction in r2
                        - DiOnAxis2:        derivative of wavefunction in r1


    '''
    def __init__(self, volume):
        '''Volume should be [[r1,r2,r12] [r1,r2,r12] ....]'''
        assert isinstance(volume,np.ndarray), 'volume should be a NumPy array'
        assert volume.shape[1]==3,"volume should be an array of dimension (nPoints,3)"
        
        self.charge = 2
        self.beta   = 0.4435
        self.volume = volume
        self.r1     = volume[:,0]
        self.r2     = volume[:,1]
        self.r12    = volume[:,2]
        self.nsamp  = self.r1.shape[0]
        self.coeff  = 0.1556
        self.norma  = 1.3891

        '''it is more convenient to have all of them calculated by separated
           and calculate each one of them only once
        '''

        # wavefunction terms
        self.fi0          = self.hydrogenic(0)
        self.fi1          = self.hydrogenic(1)
        self.xi           = self.calculateXi() 
        self.omega        = self.calculateOmega()
        self.phi          = self.calculatePhi()
        self.wavefunction = self.calculateWavefunction()

        # first derivatives
        self.DiOmegaOnAxis0 = self.calculateDiOmegaOnAxis01(0)
        self.DiOmegaOnAxis1 = self.calculateDiOmegaOnAxis01(1)
        self.DiOnAxis0      = self.calculateDiOnAxis0()
        self.DiOnAxis1      = self.calculateDiOnAxis1()
        self.DiOnAxis2      = self.calculateDiOnAxis2()
        
        # second derivatives
        self.DDOnAxes0   = self.calculateDDOnAxes0()
        self.DDOnAxes1   = self.calculateDDOnAxes1()
        self.DDOnAxes2   = self.calculateDDOnAxes2()
        self.DiD2OnAxes0 = self.calculateDiD2OnAxes0()
        self.DiD2OnAxes1 = self.calculateDiD2OnAxes1()

        # Hamilton quantities       
        self.localEnergy = self.calculateLocEnergy()
        self.contador = 0

    def __next__(self):
        self.recalculate()
        return_value = self.loss()
        return return_value

    def __iter__(self):
        return self

    def loss(self):
        '''Here you can put a scalar function for optimize'''
        self.contador+=1
        return self.localEnergy.std()

    def provideNewVolume(self, vol):
        '''if you need to change the coordinates r1, r2, r12
           you must use this method
        '''
        self.volume = vol
        self.recalculate()
        return self.localEnergy

    def recalculate(self):
        '''if some of parameters is changed by accessing directly,
           you have to execute this method in order to evaluate the loss function
           '''
        self.r1     = self.volume[:,0]
        self.r2     = self.volume[:,1]
        self.r12    = self.volume[:,2]
        self.nsamp  = self.r1.shape[0]
        # wavefunction terms
        self.fi0          = self.hydrogenic(0)
        self.fi1          = self.hydrogenic(1)
        self.xi           = self.calculateXi() 
        self.omega        = self.calculateOmega()
        self.phi          = self.calculatePhi()
        self.wavefunction = self.calculateWavefunction()

        # first derivatives
        self.DiOmegaOnAxis0 = self.calculateDiOmegaOnAxis01(0)
        self.DiOmegaOnAxis1 = self.calculateDiOmegaOnAxis01(1)
        self.DiOnAxis0      = self.calculateDiOnAxis0()
        self.DiOnAxis1      = self.calculateDiOnAxis1()
        self.DiOnAxis2      = self.calculateDiOnAxis2()
        
        # second derivatives
        self.DDOnAxes0   = self.calculateDDOnAxes0()
        self.DDOnAxes1   = self.calculateDDOnAxes1()
        self.DDOnAxes2   = self.calculateDDOnAxes2()
        self.DiD2OnAxes0 = self.calculateDiD2OnAxes0()
        self.DiD2OnAxes1 = self.calculateDiD2OnAxes1()

        # Hamilton quantities       
        self.localEnergy = self.calculateLocEnergy()

    # From this point follow the basic methods for calculating the wave function and the local energy
    # Hamilton quantities
    def calculateLocEnergy(self):
        auxiliar = (self.calculateK0() + self.calculateK1() + self.calculateK2() + self.calculateK02() + self.calculateK12() + self.calculatePotEnergy())/self.wavefunction
        return auxiliar 

    def calculatePotEnergy(self):
        return (-self.charge/self.r1 - self.charge/self.r2 + 1/self.r12)*self.wavefunction

    def calculateK0(self):
        return -0.5 * ( self.DDOnAxes0 + 2/self.r1 * self.DiOnAxis0)

    def calculateK1(self):
        return -0.5 * ( self.DDOnAxes1 + 2/self.r2 * self.DiOnAxis1)

    def calculateK2(self):
        return -( self.DDOnAxes2 + 2/self.r12 * self.DiOnAxis2)

    def calculateK02(self):
        return -self.calculateT0() * self.DiD2OnAxes0

    def calculateK12(self):
        return -self.calculateT1() * self.DiD2OnAxes1

    def calculateT0(self):
        return (self.r1**2 - self.r2**2 + self.r12**2)/(2 * self.r1 * self.r12)

    def calculateT1(self):
        return (self.r2**2 - self.r1**2 + self.r12**2)/(2 * self.r2 * self.r12)


    # second derivatives
    def calculateDDOnAxes0(self):
        '''bad performance in calculation of derivatives of omega. Pasar al constructor'''
        auxiliar =  self.norma * self.phi * (  -2* self.charge*self.DiOmegaOnAxis0 +  2 * self.coeff  ) 
        return self.charge**2 * self.wavefunction + auxiliar 

    def calculateDDOnAxes1(self):
        '''bad performance in calculation of derivatives of omega. Pasar al constructor'''
        auxiliar =  self.norma * self.phi * (  -2* self.charge*self.DiOmegaOnAxis1 +  2 * self.coeff  ) 
        return self.charge**2 * self.wavefunction + auxiliar 

    def calculateDDOnAxes2(self):
        '''Bad performance. Same pasar al constructor'''
        return -self.beta * self.DiOnAxis2

    def calculateDiD2OnAxes0(self):
        '''Bad performance. Same pasar al constructor'''
        auxiliar =  self.norma * self.fi0 * self.fi1 * np.exp(-self.beta * self.r12) / 2 * self.DiOmegaOnAxis0
        return auxiliar - self.charge * self.DiOnAxis2

    def calculateDiD2OnAxes1(self):
        ''' '''
        auxiliar =  self.norma * self.fi0 * self.fi1 * np.exp(-self.beta * self.r12) / 2 * self.DiOmegaOnAxis1
        return auxiliar - self.charge * self.DiOnAxis2

    # first derivatives
    def calculateDiOnAxis2(self):
        ''' '''
        auxiliar = self.norma * self.fi0 * self.fi1 * self.omega
        return  auxiliar * np.exp(-self.beta * self.r12) / 2


    def calculateDiOnAxis0(self):
        '''axis specifies the component. 0:r1, 1:r2'''
        return -self.charge*self.wavefunction + self.norma * self.phi *  self.DiOmegaOnAxis0

    def calculateDiOnAxis1(self):
        '''axis specifies the component. 0:r1, 1:r2'''
        return -self.charge*self.wavefunction + self.norma * self.phi *  self.DiOmegaOnAxis1

    def calculateDiOmegaOnAxis01(self, axis):
        return 2 * self.coeff * self.volume[:,axis]

    #wave function definitions

    def hydrogenic(self, axis ):
        ''' hydrogenic radial functions for n '''
        return np.exp(-self.volume[:,axis] * self.charge )

    def calculateXi(self):
        '''function XI for the correlation, it includes one parameter beta'''
        return (1 + 2*self.beta - np.exp(-self.beta * self.r12) )/(2* self.beta)

    def calculateOmega(self):
        return 1+self.coeff*(self.r1**2 + self.r2**2)

    def calculatePhi(self):
        return self.fi0*self.fi1*self.xi

    def calculateWavefunction(self):
        return self.norma * self.phi * self.omega

