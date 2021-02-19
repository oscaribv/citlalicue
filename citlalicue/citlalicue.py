import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

#import QuadraticModel from pytransit
from pytransit import QuadraticModel


class light_curve:
    """
    This class genereates light curve instances
    """


    def __init__(self,time=[],tmin=0,tmax=730,cadence=(25./60.)/60./24.):
        '''Default values correspond to a typical plato light curve'''


        #Add attributes
        self.cadence = cadence
        self.transits = False
        self.spots = False
        self.error_bar = 0
        #Create time vector, assuming the first point is taken after the first integration with integration_time
        if len(time) > 0:
            self.time = time
        else:
            self.time = np.arange(tmin,tmax,cadence)

        #Flux vector of a boring star
        self.flux = np.array([1.]*len(self.time))


    def __str__(self):
        return "This is a light curve created by the goddess Citlalicue"


    def add_transits(self,planet_parameters,planet_name='b'):

        self.params = planet_parameters
        self.transits = True

        if not hasattr(self,'flux_transits'+planet_name):

            t0 = self.params[0] #mid-transit time
            p  = self.params[1] #orbital period
            b  = self.params[2] #impact parameter
            a  = self.params[3] #stellar density
            rp = self.params[4] #scaled radius
            ldc= [self.params[5],self.params[6]] #LDC

            #Get the inclination angle from the impact parameter
            inc = np.arccos(b/a)

            #Let us use PyTransit to compute the transits
            tm = QuadraticModel(interpolate=False)
            tm.set_data(self.time)
            flux = tm.evaluate(k=rp, ldc=ldc, t0=t0, p=p, a=a, i=inc)
            #Set attribute to the class
            setattr(self,'flux_transits'+planet_name,flux)

            #self.flux = self.flux * self.flux_transits
            self.flux = self.flux * getattr(self,'flux_transits'+planet_name)


    def remove_transits(self):
        if self.transits: self.flux = self.flux / self.flux_transits
        self.transits = False


    def add_spots(self,QP=[5e-5,0.5,30.,28.]):
        """
        This attribute add stellar variability using a Quasi-periodic Kernel
        The activity is added using a george Kernel
        """

        if not hasattr(self,'flux_spots'):

            #x = np.arange(min(self.time)-1,max(self.time)+1,4./24.)
            #K = kernel_QP(x,x,QP)
            #Let us copute the covariance matrix for the GP
            #x1 = np.array(x)
            #x2 = np.array(x)
            #K = cdist(x1.reshape(len(x1),1),x2.reshape(len(x2),1))
            #QP[0] = A, QP[1] = le, QP[2] = lp, QP[3] = P
            A  = QP[0]
            le = QP[1]
            lp = QP[2]
            P  = QP[3]
            #K =  - (np.sin(np.pi*K/P))**2/2./lp**2 - K**2/2./le**2
            #K = A * np.exp(K)

            #Draw a sample from it
            #let us create the samples
            #ceros = np.zeros(len(x))
            #lc_dummy = multivariate_normal(ceros,K,size=1)
            #f = interp1d(x, lc_dummy, kind='cubic',fill_value="extrapolate")
            #light_curve = f(self.time) + 1
            #light_curve = light_curve[0]
            #self.flux_spots = light_curve

            from george import kernels, GP
            k = A * kernels.ExpSine2Kernel(gamma=1./2/lp,log_period=np.log(P)) * \
            kernels.ExpSquaredKernel(metric=le)
            gp = GP(k)
            self.flux_spots = gp.sample(self.time)

        self.flux = self.flux * self.flux_spots

        self.spots = True


    def remove_spots(self):
        if self.spots: self.flux = self.flux / self.flux_spots
        self.spots = False


    def add_white_noise(self,std=0):

        if not hasattr(self,'flux_white_noise'):

            if std == 0:
                self.error_bar = np.random.uniform(35e-6,100e-6)
            else:
                self.error_bar = std

            self.flux_white_noise = self.flux[:]
            self.flux_white_noise = np.random.normal(self.flux_white_noise,std)
            self.flux = self.flux_white_noise * self.flux


    def remove_white_noise(self):
        if self.error_bar != 0: self.flux = self.flux / self.flux_white_noise
        self.error_bar = 0


    def add_all(self,planet_parameters=[],std=0):
        self.add_transits(planet_parameters)
        self.add_spots()
        self.add_pulsations()
        self.add_white_noise(std)

    def bin_light_curve(self,tbin=10):
        from scipy.stats import binned_statistic
        tbin_days = tbin/60./24.
        bins = np.arange(min(self.time),max(self.time)+tbin_days,tbin/60./24.)
        new_t = bins + 0.5*tbin/60./24.
        new_t = new_t[:-1]
        aver = binned_statistic(self.time,self.flux,bins=bins,statistic='mean')
        new_f = aver[0]
        self.time = np.array(new_t)
        self.flux = np.array(new_f)
        self.cadence = tbin_days
        self.integration_time = tbin_days/2.


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

    def savefile(self,fname='light_curve.dat'):

        header = '#Light curve created by the goodess Citlalicue \n'
        header = '#Light curve {} \n'.format(fname)
        header += '#Span from {:4.7e} to {:4.7e} days \n'.format(min(self.time),max(self.time))
        header += '#Cadence = {:8.2f} secs \n'.format(self.cadence*24*3600)
        header += '#Variability = {} \n'.format(self.spots)
        header += '#Transits = {} \n'.format(self.transits)
        header += '#Time   Flux \n'

        big_vector = np.array([self.time,self.flux])

        np.savetxt(fname,big_vector.T,fmt='%4.7f   %4.7f',header=header)
