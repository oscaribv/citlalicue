import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

from citlalicue.citlalicue import citlali
#import QuadraticModel from pytransit
from pytransit import QuadraticModel
import pytransit.version

def bin_data(tvector,fvector,rvector,tbin=10./60/24.):
    leftt = min(tvector)
    right = leftt + tbin
    xbined = []
    fbined = []
    rbined = []
    while ( leftt < max(tvector) - tbin/2 ):
        fdummy = []
        rdummy = []
        for i in range(0,len(tvector)):
            if ( tvector[i] > leftt and tvector[i] < right ):
                fdummy.append(fvector[i])
                rdummy.append(rvector[i])
        if len(fdummy) > 1:
            fbined.append(np.mean(fdummy))
            rbined.append(np.mean(rdummy)/np.sqrt(len(rdummy)))
            #fbined.append(np.average(fdummy,weights=rdummy))
            xbined.append(leftt + tbin/2.)
        leftt = leftt + tbin
        right = right + tbin
    fbined = np.asarray(fbined)
    rbined = np.asarray(rbined)
    return np.array(xbined), np.array(fbined), np.array(rbined)


class detrend():
    """
    Ths class detrends light curves using GPs
    """

    def __init__(self,fname,tbin=10,err=0,normalise=True,star_name='Star',delimiter=' '):
        """
        Load the light curve to be detrended
        fname contains the light curve data to detrends
        tbin is the time for the creation of the bins
        err is a user input error per each datum, it has to be in the same scale as the flux
        normalise, bool, if we want the light curve to be normalised
        star_name, name of the star to analyse, this is used for the plot labels
        delimiter, delimiter of the input file, default is space, but it can be e.g., ','
        """

        #Save the name of the input file
        self.fname = fname
        #Save the name of the star
        self.star_name = star_name
        #Save the binning used
        self.tbin = tbin

        #If the error of each datum is not given, then we expect to read it in the input file
        if err == 0:
            if delimiter == ' ':
                self.time, self.flux, self.ferr = np.loadtxt(fname,unpack=True)
            else:
                self.time, self.flux, self.ferr = np.loadtxt(fname,unpack=True,delimiter=delimiter)
        else:
        #if the error is given, then let us only read time and flux, and store the ferr using the err value
            if delimiter == ' ':
                self.time, self.flux = np.loadtxt(fname,unpack=True,usecols=(0,1))
            else:
                self.time, self.flux = np.loadtxt(fname,unpack=True,usecols=(0,1),delimiter=delimiter)
            self.ferr = np.array([err]*len(self.time))

        #Normalise flux and errors
        if normalise:
            mean = np.mean(self.flux)
            self.flux = self.flux/mean
            self.ferr = self.ferr/mean

        #Create attribute with a constinuous model betweeen the minimum and maximum times
        #this is helpful to plot between gaps
        self.time_model = np.linspace(self.time.min(),self.time.max(),2500)
        self.flux_model = np.ones(len(self.time_model))

        #Create attributes with binned data each bin points
        self.time_bin, self.flux_bin, self.ferr_bin = bin_data(self.time,self.flux,self.ferr,self.tbin)

        #Number of planets, initially we have 0
        self.nplanets = 0
        self.masked_transits = False
        self.fit_planets = False

    def add_transits(self,pars,ldc):
        """
        This method includes the planets to the instance detrending
        It assumes all orbits are circular
        pars -> [T0, P, a/R*,b, Rp/R*] x Number of planets
        ldc  -> u1, u2
        """

        #Initialise the planet-related fluxes
        #this attribute contains only the planetary models for each self.time
        self.flux_planet = np.ones(len(self.time))
        #this attribute contains only the planetary models for each self.time_bin
        self.flux_planet_bin = np.ones(len(self.time_bin))
        #this attribute contains only the planetary models for each self.time_model
        self.flux_planet_model = np.ones(len(self.time_model))

        #Add parameters to the class
        #This attribute constains all the parameters for all the planets
        self.planet_pars = pars
        #this attribute constains the limb darkening coefficients following a Mandel & Agol model
        self.ldc = ldc

        #number of planets to be added
        npl = int(len(pars)/5)
        #Save the number of planets as an attribute
        self.nplanets = npl

        #import pyaneti as pti
        #for i in range(npl):
        #    incl = np.arccos(pars[3+5*i]/pars[2+5*i])
        #    ppars = [pars[0+5*i],pars[1+5*i],0,np.pi/2,incl,pars[2+5*i]]
        #    rp =  pars[4+5*i]
        #    self.flux_planet        *= pti.flux_tr_singleband_nobin(self.time,ppars,rp,ldc)
        #    self.flux_planet_bin    *= pti.flux_tr_singleband_nobin(self.time_bin,ppars,rp,ldc)
        #    self.flux_planet_model  *= pti.flux_tr_singleband_nobin(self.time_model,ppars,rp,ldc)



       #We compute the model with pytransit for self.time
        tm = QuadraticModel()
        tm.set_data(self.time)
       #We compute the model with pytransit for self.time_bin
        tm_bin = QuadraticModel()
        tm_bin.set_data(self.time_bin)
        #We compute the model with pytransit for self.time_model
        tm_model = QuadraticModel()
        tm_model.set_data(self.time_model)
        #Compute the models for all the time-series
        for i in range(npl):
            incl = np.arccos(pars[3+5*i]/pars[2+5*i])
            self.flux_planet        *= tm.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=incl,k=pars[4+5*i], ldc=ldc)
            self.flux_planet_bin    *= tm_bin.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=incl,k=pars[4+5*i], ldc=ldc)
            self.flux_planet_model *= tm_model.evaluate(t0=pars[0+5*i], p=pars[1+5*i], a=pars[2+5*i], i=incl,k=pars[4+5*i], ldc=ldc)

        #Remove the planet model from the no_planet and no_planet_bin models in order to have a light curve with no planets
        self.flux_no_planet = self.flux / self.flux_planet
        self.flux_no_planet_bin = self.flux_bin / self.flux_planet_bin
        #self.flux_no_planet_model = self.flux_model / self.flux_planet_model


    def mask_transits(self,windows=6./24.):
        '''
        Mask the transits for the binned vectors
        So the GP will work only on out of transit data
        windows -> is a list that defines the window to mask around each transit, units are given in days
        '''

        #Set an attribute saying that we have masked the transits
        self.masked_transits = True
        self.mask_window = windows

        #Extract the ephemeris from the planet_pars attribute
        if hasattr(self,'planet_pars'):
            T0 = self.planet_pars[0::5]
            P  = self.planet_pars[1::5]
        else:
            print("There are no planet parameters in the current class")

        #Check if the input windows variable is a list of a float
        #if a float, assing the same float to all the planets
        if windows.__class__ != list:
            windows = [windows]*self.nplanets
        else:
        #if a list but not enought elements, let us use the maximum windows value for all the planets
            if len(windows) != self.nplanets:
                windows = [max(windows)]*self.nplanets

        #Create a list of lists to find the regions where the transits are for each planet
        tr = [None]*self.nplanets

        #Save all the indices for each planet, where there are no transits
        for o in range(self.nplanets):
            #Compute the phase for the given planet, phase will be a list with values between 0 and 1
            #note that we are working with time_bin because time_bin is the one used for the GP analysis
            phase = ((self.time_bin-T0[o])%P[o])/P[o]
            #Transform phase to a vector with values between -0.5 and 0.5
            phase[phase>0.5] -= 1
            #Save as True all the elemets where the phase of the current planet orbit is larger than the window
            tr[o] = abs(phase) > windows[o]/P[o]
            #Each tr is a list that is True for all the elemets where we do not have transits for planet o

        #Let us combine all the trs with a logical or
        indices = tr[0]
        if self.nplanets > 1:
            for o in range(1,self.nplanets):
                indices = np.logical_and(indices,tr[o])
        #Now we have a list indices that is True for all the positions where there are any transit of any planet

        #Time to masks the transits using indices
        #Let us store the data with the planets masked out
        #We only masked them out from the *bin attributes because they are the ones to be used in the GP optimisation
        self.time_bin = self.time_bin[indices]
        self.flux_bin = self.flux_bin[indices]
        self.ferr_bin = self.ferr_bin[indices]
        self.flux_planet_bin = self.flux_planet_bin[indices]
        self.flux_no_planet_bin = self.flux_no_planet_bin[indices]


    def get_gp(self,Kernel="Exp",amplitude=1e-3,metric=10.,gamma=10.,period=10.):
        """
        Citlalicue uses the kernels provided by george,
        now the options are "Exp", "Matern32", "Matern52", and Quasi-Periodic "QP"
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

        #Save the kernel name as an attribute
        self.kernel = kernel
        #Compute the kernel with George
        self.gp = george.GP(self.kernel,mean=1)
        #We compute the covariance matrix using the binned data
        self.gp.compute(self.time_bin, self.ferr_bin)

    def draw_samples(self,nsamples=1):
        #Draw a sample of the current GP using the covariance matrix
        if hasattr(self,'gp'):
            plt.figure(figsize=(15,5))
            for i in range(nsamples):
                sample_flux = self.gp.sample(self.time_bin)
                plt.plot(self.time_bin,sample_flux,alpha=0.5)
            plt.show()
        else:
            print('You have not use the get_gp attribute yet!')

    def predict(self):
        if hasattr(self,'gp'):
            #Show the prediction of the GP with the current parameters and *bin data
            #Compute the prediction
            pred, pred_var = self.gp.predict(self.flux_no_planet_bin, self.time_bin, return_var=True)
            #Create the plot
            plt.figure(figsize=(15,5))
            plt.errorbar(self.time_bin,self.flux_bin,self.ferr_bin,fmt='o',color='k',alpha=0.25,zorder=1)
            plt.plot(self.time_bin,pred,'r',zorder=2)
            plt.show()
        else:
            print('You have not use the get_gp attribute yet!')

    #Method that computes the negative of ln(likelihood) when we have only GP with no planets
    #p has to be a vector that contains the hyper-parameters
    def neg_ln_like(self,p):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(self.flux_no_planet_bin)

    #Method that computes the negative of the gradient of ln(likelihood)
    #p has to be a vector that contains the hyper-parameters
    def grad_neg_ln_like(self,p):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(self.flux_no_planet_bin)

    #Method that computes the negative of ln(likelihood) when we have planets
    #p has to be a vector that contains the planet parameters and GP hyper-parameters
    def neg_ln_like_planet(self,p):
        #The first 5*npl elements will be planet parameters
        #The 5*npl + 1 and + 2 will be LDC
        #The last elements will be hyperparameters
        npl = self.nplanets
        self.add_transits(pars=p[0:5*npl],ldc=p[5*npl:5*npl+2])
        self.gp.set_parameter_vector(p[5*npl+2:])
        return -self.gp.log_likelihood(self.flux_no_planet_bin)

    def optimize(self,fit_planets=False):
        """
        This method optimises the model in order to find the ideal hyperparameters in order to detrend the light_curve
        fit_planets -> boolean parameter that tells if we want to fit for the planets in the optimisation or not.
                       Including the planets in the fit takes longer, but it may give better results
        """
        from scipy.optimize import minimize
        import sys
        self.fit_planets = fit_planets
        if fit_planets:
            if self.masked_transits:
                print("You cannot use fit_planets=True if you have masked the transits")
                return
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
            if not self.masked_transits: print("WARNING: You have not masked the transits and you are not fitting for the planets")
            self.result = minimize(self.neg_ln_like,self.gp.get_parameter_vector(),jac=self.grad_neg_ln_like)

        #Add the result to our GP object
        #Take the values from the optimisation
        self.gp.set_parameter_vector(self.result.x)

    def sigma_clipping(self,sigma=5):
        """
        This method performs a sigma clipping algorithm respect to the current best model
        sigma is the float that indicates the tolerance for the rejection
        this method returns the number of points eliminated
        """

        #Recompute the correlation matrix
        self.gp.compute(self.time_bin,self.ferr_bin)
        #Predict the model for the binned data set
        pred, pred_var = self.gp.predict(self.flux_no_planet_bin, self.time_bin, return_var=True)
        #Create the interpolation instance
        f = interp1d(self.time_bin, pred, kind='cubic',fill_value="extrapolate")
        self.pred = f(self.time)
        self.flux_detrended = self.flux / self.pred

        #Compute the sigma_clipping for the input flux
        #Create a vector with the resduals of the flux
        residuals = self.flux_detrended  - self.flux_planet
        #Find the indices of the data where the flux is inside the sigma limit
        indices = abs(residuals) < sigma * np.std(residuals)

        #how many points did we eliminate?
        npoints = len(self.time) - len(self.time[indices])
        print("Eliminated {} points".format(npoints))
        #plt.plot(self.time,self.flux,'ro')
        #plt.plot(self.time[indices],self.flux[indices],'ko')
        #plt.show()

        if npoints > 0:

            #Mask all the outliers using indices
            self.time = self.time[indices]
            self.flux = self.flux[indices]
            self.ferr = self.ferr[indices]

            #Recompute attributes with binned data each bin points
            self.time_bin, self.flux_bin, self.ferr_bin = bin_data(self.time,self.flux,self.ferr,self.tbin)

            self.add_transits(self.planet_pars,self.ldc)
            if self.masked_transits:
                self.mask_transits(self.mask_window)

        return npoints

    def iterative_optimize(self,sigma=5):
        """
        This method computes the optimisation iteratively using a sigma clipping algorithm
        until there are no points eliminated
        sigma is the float that indicates the tolerance for the rejection
        """
        np = 1
        it = 1
        while np > 0:
            print("Iteration {}".format(it))
            self.gp.compute(self.time_bin, self.ferr_bin)
            self.optimize()
            np = self.sigma_clipping(sigma)
            it += 1

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
            #Predict the model to be plotted
            self.flux_model_gp, self.flux_model_gp_var = self.gp.predict(self.flux_no_planet, self.time, return_var=True)
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
            #Compute the models also with interpolations
            self.flux_model_gp = f(self.time)
            f = interp1d(self.time_bin, pred_var, kind='cubic',fill_value="extrapolate")
            self.flux_model_gp_var = f(self.time)

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


    def plot(self,fsx=15,fsy=5,fname='light_curve.pdf',save=False,show=True,xlim=[None],ylim=[None],show_transit_positions=True,\
             xlabel='Time [BJD - 2,457,000]',ylabel='Normalised Flux',data_label='LC data',plot_transit_model=False,\
             tr_colors = ['#006341','#CE1126', 'b', 'k', 'y', '#ffbf00', '#ff1493'],\
             model_label='Out-of-transit Model',detrended_data_label='LC detrended',flat_model_label='Flat LC model'):
        plt.figure(figsize=(fsx,fsy))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(self.time,self.flux,'.',color="#bcbcbc",alpha=0.5,label=data_label,rasterized=True)
        if hasattr(self,'flux_model_gp'):
            plt.plot(self.time,self.flux_model_gp,'-',color="#b30000",label=model_label)
            #plt.fill_between(self.time, self.flux_model_gp - np.sqrt(self.flux_model_gp_var),
            #                 self.flux_model_gp + np.sqrt(self.flux_model_gp_var),color="#b30000", alpha=0.25)
        if hasattr(self,'flux_detrended'):
            plt.plot(self.time,self.flux_detrended-4*np.std(self.flux)-3*np.std(self.flux_detrended),
                     '.',color="#005ab3",alpha=0.5,label=detrended_data_label,rasterized=True)
            if self.fit_planets or plot_transit_model:
                plt.plot(self.time_model,self.flux_planet_model-4*np.std(self.flux)-3*np.std(self.flux_detrended),color='#ff7f00',label=flat_model_label)
            plt.ylabel('Normalised flux + offset')
        if show_transit_positions:
            if hasattr(self,'planet_pars'):
                T0 = self.planet_pars[0::5]
                P  = self.planet_pars[1::5]
                plabel = ['b','c','d','e','f','g','h','i','j','k','l','m']
                for i in range(len(T0)):
                    #where does the first transit happen?
                    n = int((T0[i] - self.time.min())%P[i]) + 100
                    t0s = np.arange(T0[i] - n*P[i],self.time.max(),P[i])
                    ys = [max(self.flux)]*len(t0s)
                    plt.plot(t0s,ys,'v',label=self.star_name+' '+plabel[i],alpha=0.75,color=tr_colors[i])
        plt.legend(loc=4,ncol=9,scatterpoints=1,numpoints=1,frameon=True)
        plt.xlim(self.time.min(),self.time.max())
        if len(xlim) == 2: plt.xlim(*xlim)
        if len(ylim) == 2: plt.ylim(*ylim)
        if save:
            plt.savefig(fname,bbox_inches='tight',dpi=300)
            plt.savefig(fname[:-3]+'png',bbox_inches='tight',dpi=300)
        if show:
            plt.show()
        plt.close()