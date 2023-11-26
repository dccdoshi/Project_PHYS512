import pandas as pd
import numpy as np
import warnings
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from more_itertools import consecutive_groups

# A class called TransitTimingVariations

# A function to get the mid-transit times (MTTs) with indices, and plot them

# A function to get the TTVs from the mid-transit times, and plot the TTVs


class TransitTimingVariations():

    def __init__(self,lc_objects,tstep,name) -> None:
        '''
        To initialize the transit-timing variation (TTV) code you need to provide the following variables:
        
        lc_objects => output of the compute_lightcurve function
        tstep => timestep used for the simulation, in *seconds*
        name => name of object to compute TTVs for, as a string, and must match the lc_objects dataframe label
        '''
        
        self.lightcurve = 1 - lc_objects[name]["BlockedFlux"]
        self.dt = tstep # seconds
        self.name = name
        self.MTTs = 'empty' # just initializing these attributes
        self.TTVs = 'empty'
        

    def compute_MTTs(self,plotfig=True):
        '''
        This function computes the mid-transit times (MTTs) of each transit by finding the middle index of each transit.
        
        INPUT:
        plotfig => set to False if you don't want the function to plot the MTT plot after computing them.

        OUTPUTS:
        MTTs => numpy array of the mid transit times, in *seconds*
        A MTT plot if requested.
        '''
        
        transits = self.lightcurve[self.lightcurve != 1].index # indices of the lightcurve data where there is blocked flux
        
        transit_groups = consecutive_groups(transits) # identify groups of consecutive indices, ie of separate transits

        final=pd.concat([self.lightcurve.iloc[i].reset_index(drop=False) for i in transit_groups],axis=1)
        
        n, m = np.shape(final)
        MTTs = np.zeros(m//2) # there are 2 columns per transit, 1 with the original index, and one with the lightcurve values
        
        for i in range(0,m,2):
            dip_inds = final.iloc[:,i] # get transit indices
            dipstart = dip_inds.iloc[0] # start of the transit
            
            endind = -1
            dipend = dip_inds.iloc[endind] # end of the transit
            while np.isnan(dipend):
                # accounts for transits that are a couple datapoints shorter than others
                endind -= 1
                dipend = dip_inds.iloc[endind]

            dipmid = (dipend+dipstart)//2 # mid-transit *index*
            dipmidalt1 = dipmid + 1
            dipmidalt2 = dipmid - 1
            dipmidalts = np.array([dipmid,dipmidalt1,dipmidalt2]) # exploring alternate midtimes in case the truncation selected wrong index
            lh = abs(dipmidalts-dipstart)
            rh = abs(dipmidalts-dipstart)
            diffs = abs(lh - rh)
            truemid_ind = np.argmin(diffs)
            truemid = dipmidalts[truemid_ind] # for most of the data, the initial dipmid ends up being the best estimate of the MTT

            MTTs[i//2] = int(truemid)*self.dt # index*dt gives the true MTT in *seconds*
            
        self.MTTs = MTTs[1:] # discard the first one as it may have started before the simulation starts
        
        # Plot the MTTs if requested
        if plotfig == True:
            plt.scatter(range(len(self.MTTs)),self.MTTs)
            plt.xlabel('Transit Number')
            plt.ylabel('Mid-transit Time [s]')
            plt.show()

        return self.MTTs


    def compute_TTVs(self,plotfig=True,plot_time_units="s"):
        '''
        This function computes TTVs from the MTTs. Uses scipy.optimize.curve_fit for linear fitting
        
        INPUTS:
        plotfig => set to False if you don't want the function to plot the MTT plot after computing them.
        plot_time_units => what units you want the TTV plot to be in. Must be "s" or "hrs".

        OUTPUT:
        TTVs => numpy array of the transit timing variations, in *seconds*.
        A TTV plot if requested.
        '''
        seconds_per_hour = 60*60
        
        # Check if the MTTs have been computed
        if type(self.MTTs) == str:
            MTTs = compute_MTTs(plotfig=False)
        
        tnum = range(len(self.MTTs)) # transit number array
        
        # Linear model to fit MTT data
        def linmodel(x,m,b):
            return m*x + b
        
        # Perform the linear fit to MTT data
        popt, pcov = curve_fit(linmodel,tnum,self.MTTs,sigma=[self.dt for mtt in self.MTTs]) # error is dt due to the discrete MTT selection process (no interpolation)
        model_rslt = linmodel(tnum,popt[0],popt[1])
        
        TTVs = self.MTTs - model_rslt # TTV is basically the residual of the MTTs linear fit (Observed-Calculated)
        self.TTVs = TTVs
        
        # Plot figures if requested
        if plotfig == True:
            # Plot the MTTs and fit
            plt.scatter(tnum,self.MTTs,label='simulated mid-transit times')
            plt.plot(tnum,model_rslt,ls='--',color='r',label='linear fit to MTTs')
            plt.xlabel('Transit Number')
            plt.ylabel('Mid-transit Time [s]')
            plt.title('Mid-transit Times for '+self.name)
            plt.legend()
            plt.show()
            
            # Plot the TTV results (basically the residuals of the MTTs
            if plot_time_units=="hrs":
                plt.plot(tnum,self.TTVs/seconds_per_hour,ls=':',marker='+',ms=10)
            else:
                plt.plot(tnum,self.TTVs,ls=':',marker='+',ms=10)
            plt.xlabel('Transit Number')
            plt.ylabel('Observed-Calculated ['+plot_time_units+']')
            plt.title('Transit Timing Variations for '+self.name)
            plt.show()
            
        return self.TTVs
