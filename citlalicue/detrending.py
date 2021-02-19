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

    def __init__(self,fname,bin=10,err=0,normalise=True,star_name='Star'):
        """Load the light curve to be detrended
        The bin variables are used to speed up computation of the GP
        """

        self.fname = fname
        self.star_name = star_name

        if err == 0:
            self.time, self.flux, self.ferr = np.loadtxt(fname,unpack=True)
        else:
            self.time, self.flux = np.loadtxt(fname,unpack=True)
            self.ferr = np.array([err]*len(self.time))

        if normalise:
            mean = np.mean(self.flux)
            self.flux = self.flux/mean
            self.ferr = self.ferr/mean

        self.time_bin = self.time[::bin]
        self.flux_bin = self.flux[::bin]
        self.ferr_bin = self.ferr[::bin]

        #Initialise the planet-related fluxes
        self.nplanets = 1
        self.flux_planet = np.ones(len(self.time))
        self.flux_planet_bin = np.ones(len(self.time_bin))
        self.flux_no_planet = self.flux
        self.flux_no_planet_bin = self.flux_bin

    def add_transits(self,pars,ldc):
        """
        Add transits:
        pars -> [T0, P, a/R*,b, Rp/R*] x Number of planets
        ldc  -> u1, u2
        """

        #Add parameters to the class
        self.planet_pars = pars
        self.ldc = ldc

        #number of planets to be added
        npl = int(len(pars)/5)
        self.nplanets = npl
        #We compute the model with
        tm = QuadraticModel()
        tm.set_data(self.time)
        tm_bin = QuadraticModel()
        tm_bin.set_data(self.time_bin)
        flux = 1
        flux_bin = 1
        for i in range(npl):
            incl = np.arccos(pars[3+5*i]/pars[2+5*i])
            flux     = flux     * tm.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=incl,k=pars[4+5*i], ldc=ldc)
            flux_bin = flux_bin * tm_bin.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=incl,k=pars[4+5*i], ldc=ldc)

        self.flux_planet = flux
        self.flux_planet_bin = flux_bin
        self.flux_no_planet = self.flux / flux
        self.flux_no_planet_bin = self.flux_bin / flux_bin


    def mask_transits(self,windows=6./24.):
        '''
        Mask the transits for the binned vectors
        So the GP will work only on out of transit data
        '''

        self.masked_transits = True

        #Extract the ephemeris from the planet_pars attribute
        if hasattr(self,'planet_pars'):
            T0 = self.planet_pars[0::5]
            P  = self.planet_pars[1::5]
        else:
            print("There are no planet parameters in the current class")


        if windows.__class__ != list:
            windows = [windows]*self.nplanets
        else:
            if len(windows) != self.nplanets:
                windows = [max(windows)]*self.nplanets

        #Create a list of lists to find the regions where the transits are for each planet
        tr = [None]*self.nplanets

        for o in range(0,self.nplanets):
            phase = ((self.time_bin-T0[o])%P[o])/P[o]
            phase[phase>0.5] -= 1
            tr[o] = abs(phase) >= (2*windows[o])/P[o]

        #Let us combine all the data with a logical or
        indices = tr[0]
        if self.nplanets > 1:
            for o in range(1,self.nplanets):
                indices = np.logical_or(indices,tr[o])

        #Let us store the data with the planets masked out
        self.time_bin = self.time_bin[indices]
        self.flux_bin = self.flux_bin[indices]
        self.ferr_bin = self.ferr_bin[indices]
        self.flux_planet_bin = self.flux_planet_bin[indices]
        self.flux_no_planet_bin = self.flux_no_planet_bin[indices]


    def get_gp(self,Kernel="Exp",amplitude=1e-3,metric=10.,gamma=10.,period=10.):
        """
        Citlalicue uses the kernels provided by george, now the options are "Exp", "Matern32", "Matern52", and Quasi-Periodic "QP"
        User can modify hyper parameters amplitude, metric, gamma, period.
        """
        import george
        from george import kernels
        if Kernel == "Matern32":
            kernel = amplitude * kernels.Matern32Kernel(metric)
        elif Kernel == "Matern52":
            kernel = amplitude * kernels.Matern52Kernel(metric)
        elif Kernel == "Exp":
            kernel = amplitude * kernels.ExpKernel(metric)
        elif Kernel == "QP":
            log_period = np.log(period)
            kernel = amplitude * kernels.ExpKernel(metric)*kernels.ExpSine2Kernel(gamma,log_period)

        self.kernel = kernel
        #Compute the kernel with George
        self.gp = george.GP(self.kernel,mean=1)
        #We compute the kernel using the binned data
        self.gp.compute(self.time_bin, self.ferr_bin)

    def draw_samples(self,nsamples=1):
        plt.figure(figsize=(15,5))
        for i in range(nsamples):
            sample_flux = self.gp.sample(self.time_bin)
            plt.plot(self.time_bin,sample_flux,alpha=0.5)
        plt.show()

    def predict(self):
        pred, pred_var = self.gp.predict(self.flux_no_planet_bin, self.time_bin, return_var=True)
        plt.figure(figsize=(15,5))
        plt.errorbar(self.time_bin,self.flux_bin,self.ferr_bin,fmt='o',color='k',alpha=0.25,zorder=1)
        plt.plot(self.time_bin,pred,'r',zorder=2)
        plt.show()

    #p has to be a vector that contains the hyper parameters
    def neg_ln_like(self,p):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(self.flux_no_planet_bin)

    def grad_neg_ln_like(self,p):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(self.flux_no_planet_bin)

    def neg_ln_like_planet(self,p):
        #The first 5*npl elements will be planet parameters
        #The 5*npl + 1 and + 2 will be LDC
        #The last elements will be hyperparameters
        npl = self.nplanets
        self.add_transits(pars=p[0:5*npl],ldc=p[5*npl:5*npl+2])
        self.gp.set_parameter_vector(p[5*npl+2:])
        return -self.gp.log_likelihood(self.flux_no_planet_bin)

    def optimize(self,fit_planets=False):
        from scipy.optimize import minimize
        import sys
        if fit_planets:
            print("Not working yet!")
            sys.exit()
            #Create guess vector including the planet parameters
            p = np.concatenate([self.planet_pars,self.ldc,self.gp.get_parameter_vector()])
            #optimise the whole likelihood optimising the planet model too
            result = minimize(self.neg_ln_like_planet,p)
            #Recompute the transits
            npl = self.nplanets
            self.add_transits(result.x[:5*npl],result.x[5*npl:5*npl+2])
            #Save the hyperparameters in self.result
            self.result = result
            self.result.x = result.x[5*npl+2]
        else:
            self.result = minimize(self.neg_ln_like,self.gp.get_parameter_vector(),jac=self.grad_neg_ln_like)

        #Add the result to our GP object
        #Take the values from the optimisation
        self.gp.set_parameter_vector(self.result.x)

    def detrend(self,method='interpolation'):
        """detrend the original data set
           There are two methods to compute the detrend light curve
           method = gp, it computes the gp for the whole data set, this is computational demanding
           method = interpolation, interpolates the GP using the GP from the binned data set, this is faster
                    but you have to be careful.
        """

        if method == 'gp':
            #Recompute the correlation matrix
            self.gp.compute(self.time,self.ferr)
            #Predict the model for the original data set
            self.pred, self.pred_var = self.gp.predict(self.flux_no_planet, self.time, return_var=True)
            #Compute the detrended flux
            self.flux_detrended = self.flux / self.pred
        elif method[0:2] == 'in':
            from scipy.interpolate import interp1d
            #Recompute the correlation matrix
            self.gp.compute(self.time_bin,self.ferr_bin)
            #Predict the model for the binned data set
            pred, pred_var = self.gp.predict(self.flux_no_planet_bin, self.time_bin, return_var=True)
            #Create the interpolation instance
            f = interp1d(self.time_bin, pred, kind='cubic',fill_value="extrapolate")
            self.pred = f(self.time)
            self.flux_detrended = self.flux / self.pred

        #Store the data in a file
        vectorsote = np.array([self.time,self.flux_detrended,self.ferr,self.flux,self.pred,self.flux_planet])
        header = "Time  Detrended_flux  flux_error  flux  GP_model  planets_model"
        fname = self.fname[:-4]+'_detrended.dat'
        print("Saving {} file".format(fname))
        np.savetxt(fname,vectorsote.T,header=header)

    def cut_transits(self,windows=6./24.):

        #Extract the ephemeris from the planet_pars attribute
        if hasattr(self,'planet_pars'):
            T0 = self.planet_pars[0::5]
            P  = self.planet_pars[1::5]
        else:
            print("There are no planet parameters in the current class")


        if windows.__class__ != list:
            windows = [windows]*self.nplanets
        else:
            if len(windows) != self.nplanets:
                windows = [max(windows)]*self.nplanets

        #Create a list of lists to find the regions where the transits are for each planet
        tr = [None]*self.nplanets

        for o in range(0,self.nplanets):
            phase = ((self.time-T0[o])%P[o])/P[o]
            phase[phase>0.5] -= 1
            tr[o] = abs(phase) <= (2*windows[o])/P[o]

        #Let us combine all the data with a logical or
        indices = tr[0]
        if self.nplanets > 1:
            for o in range(1,self.nplanets):
                indices = np.logical_or(indices,tr[o])

        #Now indices contains all the elements where there are transits
        #Let us extract them
        self.time_cut = self.time[indices]
        self.flux_cut = self.flux[indices]
        self.ferr_cut = self.ferr[indices]

        vectorsote = np.array([self.time_cut,self.flux_cut,self.ferr_cut])
        fname = self.fname[:-4]+'_cut.dat'
        if hasattr(self,"flux_detrended"):
            self.flux_detrended_cut = self.flux_detrended[indices]
            vectorsote = np.array([self.time_cut,self.flux_detrended_cut,self.ferr_cut])
            fname = self.fname[:-4]+'_detrended_cut.dat'

        print("Saving {} file".format(fname))
        np.savetxt(fname,vectorsote.T)


    def plot(self,fsx=15,fsy=5,fname='light_curve.pdf',save=False,show=True,xlim=[None],show_transit_positions=True,\
             xlabel='Time [days]',ylabel='Normalised Flux',data_label='LC data',\
             model_label='Model',detrended_data_label='LC detrended',flat_model_label='Flat LC model'):
        plt.figure(figsize=(fsx,fsy))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self.time,self.flux,'.',color="#bcbcbc",alpha=0.5,label=data_label)
        if hasattr(self,'pred'):
            plt.plot(self.time,self.pred*self.flux_planet,'-',color="#b30000",label=model_label)
        if hasattr(self,'flux_detrended'):
            plt.plot(self.time,self.flux_detrended-6*np.std(self.flux),'.',color="#005ab3",alpha=0.5,label=detrended_data_label)
            plt.plot(self.time,self.flux_planet-6*np.std(self.flux),'#ff7f00',label=flat_model_label)
            plt.ylabel('Normalised flux + offset')
        if show_transit_positions:
            if hasattr(self,'planet_pars'):
                T0 = self.planet_pars[0::5]
                P  = self.planet_pars[1::5]
                plabel = ['b','c','d','e','f','g','h','i','j','k','l','m']
                for i in range(len(T0)):
                    #where does the first transit happen?
                    n = int((T0[i] - self.time.min())%P[i])
                    t0s = np.arange(T0[i] - n*P[i],self.time.max(),P[i])
                    ys = [max(self.flux)]*len(t0s)
                    plt.plot(t0s,ys,'v',label=self.star_name+' '+plabel[i],alpha=0.75)
        plt.legend(loc=4,ncol=5,scatterpoints=1,numpoints=1,frameon=True)
        plt.xlim(self.time.min(),self.time.max())
        try:
            plt.xlim(*xlim)
        except:
            pass
        if save:
            plt.savefig(fname,bbox_inches='tight',rasterized=True)
            plt.savefig(fname[:-3]+'png',bbox_inches='tight',rasterized=True,dpi=225)
        if show:
            plt.show()
        plt.close()