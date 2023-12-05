import astropy.constants as const
import pandas as pd
import numpy as np
import warnings
import matplotlib.pylab as plt



class LightCurve():

    def __init__(self,radii_objects,rad_unit,dist_unit,xgrid) -> None:
        '''
        To initialize the lightcurve code you need to provide the following variables

        radii_objects => numpy array of the radii of the objects in the simulation in the same order as the simulation
        rad_unit => string that defines the unit for radii (can only be REarth or m)
        dist_unit => string that defines the unit used for distance in the simulation (can only be AU or m)
        xgrid => array for the time
        '''
        self.R = radii_objects

        self.R_unit = rad_unit
        if self.R_unit=="m":
            pass
        elif self.R_unit == "REarth":
            self.R = self.R * const.R_earth.value
        else:
            raise ValueError("r_unit must be 'REarth' or 'm'")


        self.dist_unit = dist_unit
        if self.dist_unit == "AU" or self.dist_unit=="m":
            pass
        else:
            raise ValueError("dis_unit must be 'AU' or 'm'")

        self.time = xgrid
        
        pass

    def compute_lightcurve(self,x,y,names):
        '''
        This function computes the lightcurve in terms of each planet.
        This code assumes that the objects are all in the same z position wher z=0.
        It considers that the observer is looking from x = +inf, y=0, therefore we consider 
        the y values for the transit and when the planet is in the +x space. 

        INPUTS:
        x => np.array that has the x positions of all the objects
        y => np.array that has the y positions of all the objects
        names => list/array of names of objects

        OUTPUT:
        objects => dict where the keys are the object names and the values are a pd table 
        that holds the X, Y, and BlockedFlux Info for the object
        '''

        ## Make sure inputs in correct format
        if len(self.R) == x.shape[1] and len(self.R) == y.shape[1]:
            pass
        else:
            raise ValueError("The number of objects in your positional array is different than the number of radii provided.")

        if self.dist_unit == "AU":
            x *= const.au.value
            y *= const.au.value

        # Make a list of pandas dataframes to deal with values 
        objects = {}
        for i in range(len(names)):

            df = pd.DataFrame({"Time":self.time,"X":x[:,i],"Y":y[:,i]})
            objects[names[i]] = df


        ## Create LightCurve Data
        # If it's the star, the blocked flux will always be zero
        objects[names[0]]["BlockedFlux"] = np.zeros(len(x))

        # Create host star variables to help distinguish between star and planet
        star = objects[names[0]]
        starR = self.R[0]

        # Find the blocked flux for each planet
        for i in range(1,len(names)):
            df = objects[names[i]]
            df["BlockedFlux"] = np.zeros(len(x))

            # Track the front and back of star and planet
            p_front = df["Y"]+self.R[i]
            p_back = df["Y"]-self.R[i]
            s_front = star["Y"]+starR
            s_back = star["Y"]-starR

            # First consider full transit scenario where the entirety of the planet is blocking the star
            full_transit = (df["X"] > 0) & (p_front<=s_front) & (p_back>=s_back)
            df.loc[full_transit,"BlockedFlux"] = (self.R[i]/starR)**2

            # Next consider partial transits where the planet is entering full transit or exiting full transit
            entering = ((p_front>=s_back) & (p_back<=s_back))
            exiting = ((p_front>=s_front) & (p_back<=s_front))
            partial_transit = (df["X"] > 0) & (entering ^ exiting)
            
            # Calculate distance between centers
            distance = np.abs(df["Y"].loc[partial_transit]-star["Y"].loc[partial_transit]).values
            # Find the blocked area
            A_intersect = self.intersection_area(distance, starR, self.R[i])
            df.loc[partial_transit,"BlockedFlux"] = (A_intersect/(4*np.pi*starR**2))

            objects[names[i]] = df

        return objects


    def intersection_area(self, d,R,r):
        '''
        Returns area between overlapping circles
        Inputs:
        d = distance between center of circles
        R = star radius
        r = planet radius
        Outputs:
        the area blocked by the planet
        '''

        f_term = r**2*np.arccos((d**2+r**2-R**2)/(2*d*r))
        s_term = R**2*np.arccos((d**2+R**2-r**2)/(2*d*R))
        t_term = -(1/2)*np.sqrt(((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))
        return f_term+s_term+t_term


    def plot_multiple_planet(self,objects,sizex,sizey, hspace, start=0,end=-1):
        '''
        This function provide a plot where you can visualize light curves caused by each planet indiviually.
        INPUTS:
        objects: a dict of the objects you want to plot, make sure the first object is the star and make sure there is more than one planet
        sizex: the x size of the figure
        sizey: the y size of the figure
        hspace: the vertical spacing with the subplots
        start: time you want to start the plot
        end: time you want to end the plot

        OUTPUT:
        A figure with subplots of each lightcurve
        '''
        
        names = list(objects.keys())[1:]

        if len(names)==1:
            raise ValueError("You need to provide more objects")

        fig, axs = plt.subplots(len(objects)-1,1,figsize=(sizex,sizey),sharex=True)

        for i in range(0,len(names)):
            axs[i].scatter(objects[names[i]]["Time"].iloc[start:end],1-objects[names[i]]["BlockedFlux"].iloc[start:end],
                            s=8,c="blue",alpha=0.7,label=names[i])
            axs[i].set_ylabel("Relative Flux")
            axs[i].legend(loc="lower right")

        axs[-1].set_xlabel("Time")
        axs[0].set_title("Light Curve of Star solely from Specific Planets")
        plt.subplots_adjust(hspace=hspace)
        return fig

    def plot_one_planet(self,object,name, sizex,sizey, start=0,end=-1):
        '''
        This function provide a plot where you can visualize light curves caused by each planet indiviually.
        INPUTS:
        object: the dataframe of just one object
        name: name of the object as a str
        sizex: the x size of the figure
        sizey: the y size of the figure
        start: time you want to start the plot
        end: time you want to end the plot

        OUTPUT:
        A figure with subplots of each lightcurve
        '''

        fig, axs = plt.subplots(1,1,figsize=(sizex,sizey))
        axs.scatter(object["Time"].iloc[start:end],1-object["BlockedFlux"].iloc[start:end],
                            s=36,c="blue",alpha=0.7,label=name)
        axs.set_xlabel("Time")
        axs.set_ylabel("Relative Flux")
        axs.set_title("Light Curve of Star solely from "+str(name))
        return fig

    def plot_total_lc(self, objects,sizex,sizey,start=0,end=-1):
        '''
        This function provide a plot where you can visualize light curves caused by each planet indiviually.
        INPUTS:
        object: the dict of all objects with star and planets
        sizex: the x size of the figure
        sizey: the y size of the figure
        start: time you want to start the plot
        end: time you want to end the plot

        OUTPUT:
        A figure with subplots of each lightcurve
        '''

        names = list(objects.keys())
        total_blocked_flux = np.zeros(len(objects[names[0]]["BlockedFlux"].iloc[start:end]))
        for i in range(len(names)):
            total_blocked_flux += objects[names[i]]["BlockedFlux"].iloc[start:end].values

        fig, axs = plt.subplots(1,1,figsize=(sizex,sizey))
        axs.scatter(objects[names[0]]["Time"].iloc[start:end],1-total_blocked_flux,
                            s=16,c="blue",alpha=0.7)
        axs.set_xlabel("Time")
        axs.set_ylabel("Relative Flux")
        axs.set_title("Light Curve of Star from Contribution of All Planets")
        return fig
        
    def get_total_lc(self, objects,start=0,end=-1):
        ''' 
        JG added this function to lightcurve.py
        Basically a copy of the code from plot_total_lc
        '''
         
          # Get total blocked flux 
        names = list(objects.keys())
        total_blocked_flux = np.zeros(len(objects[names[0]]["BlockedFlux"].iloc[start:end]))
        for i in range(len(names)):
            total_blocked_flux += objects[names[i]]["BlockedFlux"].iloc[start:end].values

        return 1-total_blocked_flux
