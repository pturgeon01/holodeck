import numpy as np
class Distribution():
    """Creates distribution objects, which store parameters and their associated dimensionality. The specific type of distribution can also be imposed, along with its respective dimensions. Dimensional reduction of given distributions is also introduced, which propagates to the distribution object's attributes. This allows for marginalization operations such as Monte Carlo Integrations and independent distribution products."""    
    def __init__(self,parameters):
            if isinstance(parameters,dict):
                self.par_names = list(parameters.keys())
                self.parameters = list(parameters.values())
            else:
                self.par_names = list(range(len(parameters)))
                self.parameters = list(parameters)
                
    @staticmethod   
    def reduce_dimension(distobj,dimension):
        i = np.where(distobj.dim_names == dimension)[0][0]
        dimensions = distobj.dimensions[:i] + distobj.dimensions[i+1:]
        if isinstance(distobj.dim_names, int):
            dim_names = list(range(len(distobj.dimensions)))
        elif (distobj.dim_names.dtype == np.dtype('<U12') or distobj.dim_names.dtype == np.dtype('<U11') or distobj.dim_names.dtype == np.dtype('<U7') ):
            if not np.shape(distobj.dim_names):
                dim_names = np.array('')
            else:
                dim_names = tuple(distobj.dim_names[:i]) + tuple(distobj.dim_names[i+1:])
                dim_names = np.array(dim_names)
        return dimensions, dim_names
   
    class PDF:
        """Creates a PDF Object for a given set of cosmological parameters, which depends on the dimensionality of the strain input"""
        def __init__(self, objects):
            self.object = objects
            self.parameters = self.object.parameters
            self.par_names = self.object.par_names
            self.log = False #Initial scale of PDF.

        def get_pdf(self, pdf, indep):
            self.distribution = np.array(pdf)
            self.independent_par = np.array(indep)
            self.dim_names = np.array(range(len(np.shape(self.distribution)) - len(self.par_names)))
            self.dimensions = np.shape(self.distribution)[len(self.par_names):]
            return self.distribution
            
        def load_pdf(self,location_pdf):
            """Loads a pre-existing pdf from a .npz file"""
            self.distribution = np.array(np.load(f'{location_pdf}/pars_{self.par_names}_pdf.npz', allow_pickle=True)['pdf'].tolist())
            self.independent_par = np.array(np.load(f'{location_pdf}/pars_{self.par_names}_pdf.npz', allow_pickle=True)['hc_ss'].tolist())
            self.dim_names = np.array(range(len(np.shape(self.distribution)) - len(self.par_names)))
            self.dimensions = np.shape(self.distribution)[len(self.par_names):]
            print(f'The pdf of the independent parameter hc_ss given the parameters {self.par_names} has been set')
            return self.distribution

            
    class Priors:
        def __init__(self, objects):
            self.object = objects
            self.parameters = self.object.parameters
            self.log = False #initial scale of priors.
            
        def Uniform(self, h, maxh = 4*10**(-13), minh = 0):
            """Creates a uniform distribution array which depends on the maximal and minimal values in the dimension \'dim\' of the pdf, and broadcasts it to the rest of the pdf."""
            if isinstance(h,float):
                if (h < maxh) and (h > minh):
                    self.distribution = 1/(maxh - minh)
                else:
                    self.distribution = np.inf

                return  self.distribution
        
        
    
    class Joint_distribution:
        def __init__(self, objects):
            self.object = objects
            self.parameters = self.object.parameters
            self.par_names = self.object.par_names
            
        def joindiag(self, dim):
            """Create a joint distribution along a given dimension, if initial distributions are independent."""
            if self.object.log != True:
                print('Converting to log-space to avoid overflow')
                self.log = True
                s = np.sum(np.log10(self.object.distribution), axis = np.where(self.object.dim_names == dim)[0][0] + len(self.par_names))
            else:
                self.log = True
                s = np.sum(self.object.distribution, axis = np.where(self.object.dim_names == dim)[0][0] + len(self.par_names))
            
            self.distribution = s
            self.dimensions, self.dim_names = Distribution.reduce_dimension(self.object, dimension = dim) 
            
            return self.distribution

    class Dimensional_Projection:
        def __init__(self, objects):
            self.object = objects
            self.parameters = self.object.parameters
            self.par_names = self.object.par_names
            self.log = self.object.log

        def project(self, dim, index):
            """Slice a number of components in a given dimension for a multi-dimensional distribution."""
            
            self.distribution = np.take(self.object.distribution, indices = index, axis = np.where(self.object.dim_names == dim)[0][0] + len(self.par_names)) #Column number is dim
            self.dimensions, self.dim_names = Distribution.reduce_dimension(self.object, dimension = dim)
            
            
            return self.distribution
    class MCI():
        """Distribution obtained from a Monte Carlo Integration."""
        def __init__(self,prior,pdf):
            self.prior = prior
            self.pdf = pdf
            self.parameters = self.pdf.parameters
            self.par_names = self.pdf.par_names
    
            
        def apply_prior(self):
            """Applies prior distribution to unmarginalized likelihood before integration to avoid product float overflow."""
            if not (self.pdf.log and self.prior.log):
                self.summand = self.pdf.distribution/self.prior.distribution
                self.log = False
            elif self.pdf.log and self.prior.log:
                self.lsummand = self.pdf.distribution - self.prior.distribution
                self.summand = np.power(10,self.lsummand)
                self.log = True
            return
    
        
        def Integration(self, dim):
            """Perform Monte Carlo Integration along a given dimension. """
            totsum = np.sum(self.summand, axis = len(self.pdf.parameters) + np.where(self.pdf.dim_names == dim)[0][0])
            totdim = self.pdf.dimensions[np.where(self.pdf.dim_names == dim)[0][0]]
            self.distribution = totsum/totdim
            self.dimensions, self.dim_names = Distribution.reduce_dimension(self.pdf, dimension = dim)
    
            return self.distribution
    
    
            
