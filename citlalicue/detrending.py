import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from citlalicue.citlalicue import light_curve
#import QuadraticModel from pytransit
from pytransit import QuadraticModel


class detrend():
    """
    Ths class detrends light curves using GPs
    """

    def __init__(self,fname,bin=10):
        """Load the light curve to be detrended
        Bin the data the speed up computations
        The bin variables are used to speed up computation of the GP
        """

        self.time_data, self.flux_data, self.ferr_data = np.loadtxt(fname,unpack=True)

        self.time_bin = self.time_data[::bin]
        self.flux_bin = self.flux_data[::bin]
        self.ferr_bin = self.ferr_data[::bin]

    def add_transits(self,pars,ldc):
        """
        Create planet model
        pars -> T0, P, a/R*,
        ldc  -> u1, u2
        """

        #Add parameters to the class
        self.planet_pars = pars
        self.ldc = ldc

        #number of planets to be added
        npl = int(len(pars)/5)
        #We compute the model with
        tm = QuadraticModel()
        tm.set_data(self.time_data)
        tm_bin = QuadraticModel()
        tm_bin.set_data(self.time_bin)
        flux = 1
        flux_bin = 1
        for i in range(npl):
            flux     = flux     * tm.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=pars[3+5*i],k=pars[4+5*i], ldc=ldc)
            flux_bin = flux_bin * tm_bin.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=pars[3+5*i],k=pars[4+5*i], ldc=ldc)

        self.flux_data_planet = flux
        self.flux_bin_planet = flux_bin
        self.flux_no_planet = self.flux_data / flux
        self.flux_no_planet_bin = self.flux_bin / flux_bin


    def create_gp(self,Kernel="Exp"):
        import george
        from george import kernels
        if Kernel == "Matern32":
            kernel = 0.1 * kernels.Matern32Kernel(10.)
        elif Kernel == "Matern32":
            kernel = 0.1*kernels.Matern32Kernel(10.)
        elif Kernel == "Exp":
            kernel = 0.1*kernels.ExpKernel(10.)

        self.kernel = kernel
        #Compute the kernel with George
        self.gp = george.GP(self.kernel,mean=1)
        #We compute the kernel using the binned data
        self.gp.compute(self.time_bin, self.ferr_bin)

    def draw_sample(self):
        sample_flux = self.gp.sample(self.time_bin)
        plt.plot(self.time_bin,sample_flux)
        plt.show()

    def predict(self):
        self.pred, self.pred_var = self.gp.predict(self.flux_no_planet_bin, self.time_bin, return_var=True)
        plt.plot(self.time_bin,self.flux_bin,'ko',alpha=0.25)
        plt.plot(self.time_bin,self.pred,'r')
        plt.show()


    #p has to be a vector that contains the planet parameters + the hyper parameters
    #def neg_ln_like(p,t,f,npl):
    def neg_ln_like(self,p):
      #The first 5*npl elements will be planet parameters
     #The 5*npl + 1 and + 2 will be LDC
     #The last elements will be hyperparameters
        #    f_local = f - transits(t,p[0:5*npl],p[5*npl:5*npl+2],npl)
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(self.flux_no_planet_bin)

    #p has to be a vector that contains the planet parameters + the hyper parameters
    #def grad_neg_ln_like(p,t,f,npl):
    def grad_neg_ln_like(p):
        #The first 5*npl elements will be planet parameters
        #The 5*npl + 1 and + 2 will be LDC
        #The last elements will be hyperparameters
    #    f_local = f - transits(t,p[0:5*npl],p[5*npl:5*npl+2],npl)
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(self.flux_no_planet_bin)

    def optimize(self):
        from scipy.optimize import minimize
        self.result = minimize(self.neg_ln_like,self.gp.get_parameter_vector())

    def detrend(self):
        """detrend the original data set"""
        #Take the values from the optimisation
        self.gp.set_parameter_vector(self.result.x)
        #Recompute the correlation matrix
        self.gp.compute(self.time_data,self.ferr_data)
        #Predict the model for the original data set
        pred, pred_var = self.gp.predict(self.flux_no_planet, self.time_data, return_var=True)
        #Compute the detrended flux
        self.flux_detrended = self.flux_data / pred

    def cut_transits(self,duration=6./24.):
        T0 = self.planet_pars[0::5]
        P  = self.planet_pars[1::5]

        #Let us create a vector that includes all the data containing all the transits
        nplanets = len(periods)

        tr = [None]*nplanets

        for o in range(0,nplanets):
            phase = ((t-T0[o])%periods[o])/periods[o]
            phase[phase>0.5] -= 1
            tr[o] = abs(phase) <= (2*durations[o])/periods[o]

        indices = tr[0]
        if nplanets > 0:
            for o in range(1,nplanets):
                indices = np.logical_or(indices,tr[o])

    def __str__(self):
        return "This is a light curve"

    def plot(self,fsx=8,fsy=8/1.618,fname='light_curve.pdf',jump=1,save=False,show=True,xlim=[None]):
        plt.figure(figsize=(fsx,fsy))
        plt.plot(self.time[::jump],self.flux[::jump],'.',label='LC model')
        plt.xlabel('Time [days]')
        plt.ylabel('Normalised flux')
        plt.legend(loc=1,ncol=5,scatterpoints=1,numpoints=1,frameon=True)
        try:
            plt.xlim(*xlim)
        except:
            pass
        if save:
            plt.savefig(fname,bbox_inches='tight')
        if show:
            plt.show()
        plt.close()