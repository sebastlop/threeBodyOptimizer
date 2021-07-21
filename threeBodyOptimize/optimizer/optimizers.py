import numpy as np


class optimize:
    '''busqueda al azar para un conjunto de parametros. 
       *params should be string refference to obj attributes to be optimized
       and lower and upper limit to be sampled
       obj should have a method named loss for optimization
       obj must be an iterator, or have __iter__ method defined
    '''
    def __init__(self, obj, **params):
        assert 'loss' in dir(obj), 'El objeto no tiene el metodo loss'
        self.obj        = obj
        self.optimizable= []
        self.lb         = []
        self.ub         = []
        for key in params.keys():
            if(params[key][0] in dir(obj)):
                print('parameter= (' + key +" : "+params[key][0] + ") will be optimized" )
                self.optimizable.append(params[key][0])
                self.lb.append(params[key][1])
                self.ub.append(params[key][2])
            else:
                print('parameter= (' + key +" : "+params[key][0] + ") does not exist in you object" )
                print('***** Fix it! *****')
                raise ValueError
        print('optimizing: ', self.optimizable)
        self.nDim   = len(self.lb)
        self.history ={}



    def simpleRandomSerach(self, epochs=1, nSamples=100, center = [1,], radii = [1,]):
        '''Simple Random search algorithm samples with random nSamples points for each parameter 
           to be optimized in the interval [center-radii, center+radii]. Uses a bubble method to store the minimum '''
        assert len(radii) == self.nDim, 'please provide same dimension than number of for radii and center'
        self.history['optimizer'] = 'simpleRandomSerach'
        center = np.array(center)
        radii = np.array(radii)
        self.history['loss'] = []

        mini = np.inf
        sMin = 0
        
        for e in range(epochs):
            samples = np.random.uniform(center-radii,center+radii,size=(nSamples,self.nDim))

            for s in samples:
            
                for i in range(self.nDim):
                    setattr(self.obj, self.optimizable[i] , s[i])
            
                next(self.obj)
                if mini > self.obj.loss():  #bubble method. Bad performance
                    mini = self.obj.loss()
                    sMin = s
            
            self.history['loss'].append( np.array( [sMin.flatten(), mini] ) )
            print(f'epoch: {e} - loss:{round(mini,5)} - at: {[round(i,5) for i in sMin]}')
            #redefine center at minimum
            center = sMin

        return np.array( [sMin.flatten(), mini] ) 


    def variableStepRandomSerach(self, epochs=1, nSamples=100, center = [1,], radii = [1,]):
        '''This implementation of the Random search algorithm samples with uniform random nSamples points for each parameter 
           to be optimized in the interval [center-radii, center+radii]. It stores a list with function values and redefine the radious
           as the difference between two lower values of the function.
           THIS METHOD DOES NOT GUARANTEE ANY CONVERGENCE TO A GLOBAL MINIMUM'''
        assert len(radii) == self.nDim, 'please provide same dimension than number of for radii and center'
        self.history['optimizer'] = 'simpleRandomSerach'
        center = np.array(center)
        radii = np.array(radii)
        self.history['loss'] = []

        
        for e in range(epochs):
            samples = np.random.uniform(center-radii,center+radii,size=(nSamples,self.nDim))

            mini = []

            for s in samples:
            
                for i in range(self.nDim):
                    setattr(self.obj, self.optimizable[i] , s[i])
            
                next(self.obj)

                mini.append(self.obj.loss())

# TRY ONE with argmin ... TRY PERFORMANCE IT BY SORTING LIST!!!!
            #with the list of lossess we pick the lowest value and set the center at this value
            idx = np.argsort(mini)
            center = samples[idx[0]]
            radii = np.abs(center - samples[idx[1]])
            #save the history
            self.history['loss'].append( np.array( [samples[idx[0]].flatten(), mini[idx[0]]] ) )
            print(f'epoch: {e} - loss:{round(mini[idx[0]],5)} - at: {[round(i,5) for i in center]} ]')

            if radii.all()<1e-7:
                print('convergence reached. radii:',radii)
                break

        return np.array( [samples[idx[0]].flatten(), mini[idx[0]] ] ) 

    
class volumeSampler:
    '''This class allow to pick distribution over the volume r1 r2 r12 based on relevant regions
    of the space depending on cheap approximations like single Z variational wavefunctions'''
    def __init__(self, nSamples):
        self.nSamples = nSamples
    
    def singleZDistribution(self):
        r1 = np.random.exponential(1/1.6875, self.nSamples)
        r2 = np.random.exponential(1/1.6875, self.nSamples)
        r12 = (r1+r2)- np.abs(np.random.normal(0,(r1+r2-np.abs(r1-r2))/4, self.nSamples))
        return np.vstack([r1,r2,r12]).T