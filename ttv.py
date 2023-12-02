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

    def __init__(self,lc_objects,dt,name) -> None:
        '''
        To initialize the transit-timing variation (TTV) code you need to provide the following variables:
        
        lc_objects => output of the compute_lightcurve function
        dt => this is the error given to the linear regression fit
        name => name of object to compute TTVs for, as a string, and must match the lc_objects dataframe label
        '''
        
        self.lightcurve = 1 - lc_objects[name]["BlockedFlux"]
        self.time = lc_objects[name]["Time"] # seconds
        self.name = name
        self.MTTs = 'empty' # just initializing these attributes
        self.TTVs = 'empty'
        self.dt = dt
        

    def compute_MTTs(self,plotfig=True,method="index"):
        '''
        This function computes the mid-transit times (MTTs) of each transit by finding the middle index of each transit.
        
        INPUT:
        plotfig => set to False if you don't want the function to plot the MTT plot after computing them.
        method => which method to use to extract mid-transit times.

        OUTPUTS:
        MTTs => numpy array of the mid transit times, in *seconds*
        A MTT plot if requested.
        '''
        
        transits = self.lightcurve[self.lightcurve != 1].index # indices of the lightcurve data where there is blocked flux
        
        transit_groups = consecutive_groups(transits) # identify groups of consecutive indices, ie of separate transits

        final=pd.concat([self.lightcurve.iloc[i].reset_index(drop=False) for i in transit_groups],axis=1)
        
        n, m = np.shape(final)
        MTTs = np.zeros(m//2) # there are 2 columns per transit, 1 with the original index, and one with the lightcurve values
        
        if method == "interpolate":
            # Alternate method, interpolation
            for i in range(0,m,2):
                dip_inds = final.iloc[:,i] # get transit indices

                if i==0:
                    dipstart = dip_inds.iloc[0]   # index of start of transit (we take the point right before the flux starts to be blocked)
                else:
                    dipstart = dip_inds.iloc[0]-1

                endind = -1
                dipend = dip_inds.iloc[endind] # index of end of transit
                while np.isnan(dipend):
                    # accounts for transits that are a couple datapoints shorter than others
                    endind -= 1
                    dipend = dip_inds.iloc[endind]

                dipend += 1 # (we take the point right after the last point where there is blocked flux)
                midtime = (self.time[dipend]+self.time[dipstart])/2 # MTT, "index"*dt gives the true MTT in *seconds*
                MTTs[i//2] = midtime
        
        elif method == "index":
            # middle index method
            for i in range(0,m,2):
                dip_inds = final.iloc[:,i] # get transit indices
                dipstart = dip_inds.iloc[0] - 1 # start of the transit

                endind = -1
                dipend = dip_inds.iloc[endind] # end of the transit
                while np.isnan(dipend):
                    # accounts for transits that are a couple datapoints shorter than others
                    endind -= 1
                    dipend = dip_inds.iloc[endind]
                    
                dipend += 1

                dipmid = (dipend+dipstart)//2 # mid-transit *index*
                dipmidalt1 = dipmid + 1
                dipmidalt2 = dipmid - 1
                dipmidalts = np.array([dipmid,dipmidalt1,dipmidalt2]) # exploring alternate midtimes in case the truncation selected wrong index
                lh = abs(dipmidalts-dipstart)
                rh = abs(dipmidalts-dipend)
                diffs = abs(lh - rh)
                truemid_ind = np.argmin(diffs)
                truemid = dipmidalts[truemid_ind] # for most of the data, the initial dipmid ends up being the best estimate of the MTT

                MTTs[i//2] = self.time[int(truemid)] # index*dt gives the true MTT in *seconds*
        
        else: 
            raise ValueError("MTT extraction method must be either 'interpolate' or 'index'.")
        
            
        self.MTTs = MTTs[1:] # discard the first one as it may have started before the simulation starts
        
        # Plot the MTTs if requested
        if plotfig == True:
            plt.scatter(range(len(self.MTTs)),self.MTTs)
            plt.xlabel('Transit Number')
            plt.ylabel('Mid-transit Time [s]')
            plt.show()

        return self.MTTs


    def compute_TTVs(self,plotfig=True,plot_time_units="min",method="index",sizex=25,sizey=10,text_size=10,y_lim_min=20):
        '''
        This function computes TTVs from the MTTs. Uses scipy.optimize.curve_fit for linear fitting
        
        INPUTS:
        plotfig => set to False if you don't want the function to plot the MTT plot after computing them.
        plot_time_units => what units you want the TTV plot to be in. Must be "s", "min" or "hrs".
        method => which method to use to extract mid-transit times.
        sizex => the x size of the figure
        sizey => the y size of the figure
        text_size => the size of caption text in the figure
        y_lim => for the "min" TTV plot only, axis limits in the y direction are set to +-y_lim

        OUTPUT:
        TTVs => numpy array of the transit timing variations, in *seconds*.
        A TTV plot if requested.
        '''
        seconds_per_hour = 60*60
        seconds_per_min = 60
        
        # Check if the MTTs have been computed
        if type(self.MTTs) == str:
            MTTs = compute_MTTs(plotfig=False,method=method)
        
        tnum = range(len(self.MTTs)) # transit number array
        
        # Linear model to fit MTT data
        def linmodel(x,m,b):
            return m*x + b
        
        # Perform the linear fit to MTT data
        popt, pcov = curve_fit(linmodel,tnum,self.MTTs,sigma=[self.dt for mtt in self.MTTs]) # take timestep as the error
        model_rslt = linmodel(tnum,popt[0],popt[1])
        print('slope= ',popt[0],' intercept= ',popt[1])
        print('Errors=',np.sqrt(np.diag(pcov)))
        
        TTVs = self.MTTs - model_rslt # TTV is basically the residual of the MTTs linear fit (Observed-Calculated)
        self.TTVs = TTVs
        
        # Plot figures if requested
        if plotfig == True:
            # Plot the MTTs and fit
            plt.rcParams.update({'font.size': 10})
            plt.scatter(tnum,self.MTTs,label='simulated mid-transit times')
            plt.plot(tnum,model_rslt,ls='--',color='r',label='linear fit to MTTs, $T_o=%.1f$ s, $P=%.1f$ s'%(popt[0],popt[1]))
            plt.xlabel('Transit Number')
            plt.ylabel('Mid-transit Time [s]')
            plt.title('Mid-transit Times for '+self.name)
            plt.legend()
            plt.show()
            
            # Plot the TTV results (basically the residuals of the MTTs)
            plt.figure(figsize=(sizex,sizey))
            plt.rcParams.update({'font.size': text_size})
            if plot_time_units=="hrs":
                plt.plot(tnum,self.TTVs/seconds_per_hour,ls=':',marker='+',ms=10)
            if plot_time_units=="min":
                plt.plot(tnum,self.TTVs/seconds_per_min,ls=':',marker='+',ms=10)
                plt.ylim((-y_lim_min,y_lim_min))
            elif plot_time_unit=="min":
                plt.plot(tnum,self.TTVs,ls=':',marker='+',ms=10)
            else: 
                raise ValueError("plot_time_units must be either 's', 'min' or 'hrs'.")
            
            plt.xlabel('Transit Number')
            plt.ylabel('Observed-Calculated ['+plot_time_units+']')
            plt.title('Transit Timing Variations for '+self.name)
            plt.show()
            
            plt.rcParams.update({'font.size': 10})
            
        return self.TTVs