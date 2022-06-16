''' Spin evolution file for solid body evolution. Can evolve a single star's
    rotation rates in time, or entire ensembles of stars. If the latter, can
    also construct probability densities using brute force or deterministic
    methods.

    Code is often used in combination with other files (such as tau squared and
    MCMC codes) for plotting or parameter minimisation. To run, this code needs:

    control/constants script to control the:
        1) rotational evolution code
        2) gridding of model probability densities
  '''
# =============================import codes=====================================

# import python packages
import sys
import numpy as np
import scipy
import math
import time
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# import my code
sys.path.insert(0, '../data')
import myconstants as const
'''=========================stellar stucture setup==========================='''
# =====reading in stellar structure models and creating stellar dictionary======

def read_Baraffe2015_models(mass):
    #Reads in Baraffe stellar evolution models for a specified mass
    #in:
    #   mass: float value of which mass is desired          [MSun]
    #out:
    #   time: array of time from Baraffe models             [Myr]
    #   inertia: array of moment of inertia                 [gcm^2]
    #   turnoverTime: array of convective turnover times    [TauSun]
    #   stellarRadius: radius of the star                   [RSun]
    #   Teff: effective temperature                         [K]
    #   massCore: mass of radiative core                    [MSun]
    #   radiusCore: radius of radiative core                [RSun]
    #   k2conv: constant required for inertia of envelope
    #   k2rad: constant required for inertia of core

    # Reading in the data from relevant file
    f = open('../data/BHAC15_tracks+structure.txt')
    lines = f.readlines()[44:]
    f.close()

    #Filling lists from Baraffe data file
    logt = []
    Teff = []
    stellarRadius = []
    massCore = []
    radiusCore = []
    k2conv = []
    k2rad = []

    #Filling lists for desired mass
    for line in lines:
        data = line.split()
        if line[0] != '!':
            if line == '\n':
                continue

            if float(data[0]) == mass:
                logt.append(float(data[1]))                     #[yr]
                Teff.append(float(data[2]))                     #[K]
                stellarRadius.append(float(data[5]))            #[RSun]
                massCore.append(float(data[9]))                 #[MSun]
                radiusCore.append(float(data[10]))              #[RSun]
                k2conv.append(float(data[11]))
                k2rad.append(float(data[12]))

    #===Calculating moment of inertia from Baraffe models===
    #Converting lists to np arrays
    k2conv = np.array(k2conv)
    k2rad = np.array(k2rad)
    stellarRadius = np.array(stellarRadius)
    Teff = np.array(Teff)
    #Moment of inertia in kg*m^2 (radius of star changes)
    inertia=(k2conv**2+k2rad**2)*mass*(const.MSun)*(stellarRadius*const.RSun)**2
    # Converting to cgs units
    inertia = inertia*1000.*10.**4 #In g*cm^2
    #Baraffe time in Myr
    time = np.power(10,logt)/1000000.
    #Calculating convective turnover time from Cranmer & Saar 2011 eq 36.
    turnoverTime = 314.24*np.exp(-(Teff/1952.5)-(Teff/6250.)**18)+0.002
    #Convert to solar values
    turnoverTime = turnoverTime/const.TauSun

    return time,inertia,turnoverTime,stellarRadius,Teff,massCore,\
           radiusCore,k2conv,k2rad

def get_spline(x,y):
    # gets spline coefficients for input arrays.
    # in:
    #   x: x array of data e.g. time
    #   y: y array of data corresponding to x (e.g. radius)
    # out:
    #  y2: fitted spline

    n = len(x) #   n: number of elements in x (and y) array

    # fill arrays with zeros
    y1 = np.zeros(n)
    y2 = np.zeros(n)
    u = np.zeros(n)

    n = n-1
    i=1
    while i < n :
        dx1 = x[i]-x[i-1]
        dx2 = x[i+1]-x[i]
        dx3 = x[i+1]-x[i-1]
        sig = dx1/dx3
        p = sig*y2[i-1]+2
        y2[i] = (sig-1)/p
        if x[i]==x[i-1] :
            a1 = dx2/dx3
            a2 = (dx2-dx1)/dx2
            a3 = 0
            u[i] = (6*(y[i+1]-y[i])/dx2)/dx3 - sig*u[i-1]/p
        else :
            a1 = -dx2/(dx3*dx1)
            a2 = (dx2-dx1)/(dx2*dx1)
            a3 = dx1/(dx2*dx3)
            u[i] = 6*( (y[i+1]-y[i])/dx2 - (y[i]-y[i-1])/dx1 )/dx3-sig*u[i-1]/p
        y1[i] = a1*y[i-1]+a2*y[i]+a3*y[i+1]
        i += 1
    y2[n] = 0
    i = n-1
    while i>0:
        y2[i]=y2[i+1]*y2[i]+u[i]
        i -= 1
    return y2

class stellarModel:
    # stores all stellar structure information for each mass of star, as well
    # as spline fits to each parameter vs time. From this, the parameter of any
    # star can be obtained for use in the rotational evolution code. Once spin
    # evolution is calculated, it is stored here along with stellar parameters.
    # Similarly, the probability densities rho can be stored in the class.

    # Initializer / Instance Attributes
    def __init__(self, mass, stellarModel='Baraffe2015'):

        # spin evolution of brute force method
        self.spinEvolOmega = []
        self.spinEvolTime = []
        self.spinEvolOmegaSpline = []

        # deterministic evolution for ONE age:
        self.Pi = []
        self.Pf = []
        self.dPi_dPf = []
        self.drho_dPf = []

        self.index = 0 # keep track of index for interpolation

        # stellar structure
        if stellarModel == 'Baraffe1998': # Don't use Baraffe1998 anymore...
            stellarModels = readBaraffe1998Models(mass)
        if stellarModel == 'Baraffe2015':
            stellarModels = read_Baraffe2015_models(mass)

        self.massStar = mass
        self.time = np.array(stellarModels[0],\
                                        dtype='double')   # model time [Myr]
        self.inertiaStar = np.array(stellarModels[1],\
                                        dtype='double')   # [gcm^2]
        self.turnoverTime = np.array(stellarModels[2],\
                                        dtype='double')   # [Tau Sun]
        self.radiusStar = np.array(stellarModels[3],\
                                        dtype='double')   # [RSun]
        self.Teff = np.array(stellarModels[4],\
                                        dtype='double')   # [K]
        self.massCore = np.array(stellarModels[5],\
                                        dtype='double')   # MSun
        self.radiusCore = np.array(stellarModels[6],\
                                        dtype='double')   # RSun
        self.constEnv = np.array(stellarModels[7],\
                                        dtype='double')
        self.constCore = np.array(stellarModels[8],\
                                        dtype='double')

        # calculating moments of inertia of core and envelope
        self.inertiaCore = np.array(self.constCore**2*(self.massStar*const.MSun)
                           *(self.radiusStar*const.RSun)**2,
                           dtype='double') #[gcm^2]
        self.inertiaEnv  = np.array(self.constEnv**2*(self.massStar*const.MSun)
                           *(self.radiusStar*const.RSun)**2,
                           dtype='double')  #[gcm^2]

        # calculating dI/dt and dM/dt for entire star, core and envelope
        alldI_dt  = []
        alldIc_dt = []
        alldIe_dt = []
        Mdot = []
        for i in range(len(self.time)-1):

            alldI_dt.append( (self.inertiaStar[i+1] - self.inertiaStar[i]) /
                    ((self.time[i+1]-self.time[i]) *const.Myr)) #[g cm^2/s]
            alldIc_dt.append( (self.inertiaCore[i+1] - self.inertiaCore[i])/
                    ((self.time[i+1]-self.time[i]) *const.Myr)) #[g cm^2/s]
            alldIe_dt.append( (self.inertiaEnv[i+1] - self.inertiaEnv[i])  /
                    ((self.time[i+1]-self.time[i]) *const.Myr)) #[g cm^2/s]
            Mdot.append( (self.massCore[i+1]-self.massCore[i]) /
                    (self.time[i+1]-self.time[i]) * const.MSun/const.Myr)#[kg/s]

            # just fill the last value into the end of the array
            if i == len(self.time) - 2:
                Mdot.append(Mdot[-1])
                alldI_dt.append(alldI_dt[-1])
                alldIe_dt.append(alldIe_dt[-1])
                alldIc_dt.append(alldIc_dt[-1])

        self.dIdtStar = np.array(alldI_dt,dtype='float')
        self.dIdtCore = np.array(alldIc_dt,dtype='float')
        self.dIdtEnv = np.array(alldIe_dt,dtype='float')
        self.Mdot = np.array(Mdot)

        # interpolate all known stellar parameters
        self.inertiaStarSpline = get_spline(self.time,self.inertiaStar)
        self.turnoverTimeSpline = get_spline(self.time,self.turnoverTime)
        self.radiusStarSpline = get_spline(self.time,self.radiusStar)
        self.TeffSpline = get_spline(self.time,self.Teff)
        self.massCoreSpline = get_spline(self.time,self.massCore)
        self.radiusCoreSpline = get_spline(self.time,self.radiusCore)
        self.inertiaCoreSpline = get_spline(self.time,self.inertiaCore)
        self.inertiaEnvSpline = get_spline(self.time,self.inertiaEnv)
        self.dIdtStarSpline = get_spline(self.time,self.dIdtStar)
        self.dIdtEnvSpline = get_spline(self.time,self.dIdtEnv)
        self.dIdtCoreSpline = get_spline(self.time,self.dIdtCore)
        self.MdotSpline = get_spline(self.time,self.Mdot)

        # storing the above parameters in a dictionary
        self.parameters = {
            'radiusStar' : self.radiusStar,
            'Teff': self.Teff,
            'inertiaStar' : self.inertiaStar,
            'turnoverTime'  : self.turnoverTime,
            'massCore' : self.massCore,
            'radiusCore' : self.radiusCore,
            'inertiaCore' : self.inertiaCore,
            'inertiaEnv' : self.inertiaEnv,
            'dIdtEnv' : self.dIdtEnv,
            'dIdtCore' : self.dIdtCore,
            'dIdtStar': self.dIdtStar,
            'Mdot' : self.Mdot,
            'spinEvolOmega' : self.spinEvolOmega,
            'spinEvolTime' : self.spinEvolTime,

            'radiusStarSpline' : self.radiusStarSpline,
            'TeffSpline' : self.TeffSpline,
            'inertiaStarSpline' : self.inertiaStarSpline,
            'turnoverTimeSpline'  : self.turnoverTimeSpline,
            'massCoreSpline' : self.massCoreSpline,
            'radiusCoreSpline' : self.radiusCoreSpline,
            'inertiaCoreSpline' : self.inertiaCoreSpline,
            'inertiaEnvSpline' : self.inertiaEnvSpline,
            'dIdtEnvSpline' : self.dIdtEnvSpline,
            'dIdtCoreSpline' : self.dIdtCoreSpline,
            'dIdtStarSpline': self.dIdtStarSpline,
            'MdotSpline' : self.MdotSpline}


    def interpSpline(self, stellarParameter, xnew, istart=0) :
        # interpolation function using a spline
        # in:
        #   x: array of e.g. times
        #   y: keyword of desiered array e.g. 'radiusStar' will map to array of
        #       radii
        #   xnew: the time for which you want info
        #   istart: for if you want to keep track of the integers
        #   y2: the coefficients from the spline
        # out:
        #   interpolated value
        #   iend: integer from which to start next interpolation
        x = self.time
        y2 = self.parameters['%sSpline' %stellarParameter]
        y = self.parameters[stellarParameter]

        # start at a given integer
        i = istart
        i = SO.index

        # find the value that you want to interpolate between xnew.
        while x[i] < xnew :
            i = i+1

        # keep track of this integer:
        iend = i

        # then just interpolate between these two values
        hi = (x[i]-x[i-1])
        a = (x[i]-xnew)/hi
        b = (xnew-x[i-1])/hi

        return a*y[i-1]+b*y[i]+((a**3-a)*y2[i-1]+(b**3-b)*y2[i])*hi**2/6, iend

    def LinearInterpolate(self,t,stellarParameter):

        #linearly interpolate to find one interpolated value for singular time
        # for a given mass
        #In:
        #   t: The time for which the corresponding interpolated value is
        #       found [Myr]
        #   self.time: Baraffe model times corresponding to certain mass [Myr]
        #   self.dIdtStar: Baraffe array corresponding to certain mass
        #                   [Units depend on what array is interpolated]
        #Out:
        #   I_interpolated: the moment of inertia corresponding to
        #                   time t [g*cm^2]
        #   t2: The second time used to interpolate the moment of inertia [Myr]
        #Notes:
        #   I have checked edge cases!

        stellarParameter = self.parameters[stellarParameter]


        #Find closest time and index i in Baraffe time [Myr] array
        diff = abs(t - self.time)
        i = np.argmin(diff)
        #If the closest Baraffe time is not greater than the input time:
        if t - self.time[i] < 0:
            i = i - 1
        t1 = self.time[i] #closest time
        #Finding second time to interpolate between
        if t < self.time[len(self.time)-1]:
            t2 = self.time[i+1]
            #linearly interpolate between t1 and t2
            y0 = stellarParameter[i]
            y1 = stellarParameter[i+1]
            x0 = t1
            x = t
            x1 = t2
            Interpolated = y0 + (x - x0)*(y1-y0)/(x1-x0)
        #For cases self.time time is the last Baraffe time or greater
        elif t == self.time[len(self.time)-1]:
            Interpolated = stellarParameter[i]
        else:
            print('CAUTION: end of Baraffe models reached, cannot interpolate')
            return
        return Interpolated, t2

    def param_at_time(self,stellarParameter,xnew,interp_method='linearSpline'):
        if interp_method == 'linearSpline':
            interpolatedValue = self.interpSpline(stellarParameter,xnew)
        if interp_method == 'linear':
            interpolatedValue = self.LinearInterpolate(xnew,stellarParameter)
        if stellarParameter == 'dIdtStar':
            interpolatedValue = self.LinearInterpolate(xnew,stellarParameter)

        return interpolatedValue

    def interpolateSpinEvolution(self,age,mass,tcX,tcY):

        period = []
        omegaarr = []
        for i in range(const.NInitialConditions):
            # linear interpolate the spin evolution
            omega = self.LinearInterpolate(age,self.spinEvolOmega[i],
                    np.array(self.spinEvolTime[i]),mass,
                    tcX,tcY)[0]
            omegaarr.append(omega)
            period.append(2.*const.pi/(omega*const.OmegaSun)/const.day)

        return period

    def getRhoSpline(self,Pf):
        # Pf can either be an array or a singular value.

        # Check if Pf is an array
        if isinstance(Pf,(list,np.ndarray)) == True:
            rho = []
            # if so, cycle through each Pf and find corresponding rho
            for i in range(len(Pf)):
                rho.append(self.PfPi_univariateSpline.derivatives(Pf[i])[1]/
                           self.PiPfInterpolated(Pf[i])*self.Matt2020norm)
            self.rho = rho

            # check if rho is ever less than or equal to zero
            indices = np.where(np.array(rho) <= 0.)[0]
            if len(indices) > 0:
                # re-evaluate spline for more points if it goes below zero
                rho = ['re-evaluate',rho,indices]
        else:
            # assumes Pf is a singular value
            rho = self.PfPi_univariateSpline.derivatives(Pf)[1]/\
                  self.PiPfInterpolated(Pf)*self.Matt2020norm

        return rho

    def interpolate_tauc(self):
        # fits a function to turnoverParametersX and turnoverParametersY
        # to make interpolation quicker

        temp_x = []
        temp_y = []

        for i in range(len(self.turnoverParametersX)):
            temp_x.append(self.turnoverParametersX[i])
            temp_y.append(self.turnoverParametersY[i])
        temp_x.append(self.endknot[0])
        temp_y.append(self.endknot[1])

        linearFit = scipy.interpolate.interp1d(temp_x,temp_y,kind='linear')

        return linearFit

# storing all structure info for each mass in a dictionary
stellarDictionary = {
                    '0.1' : stellarModel(0.1,'Baraffe2015'),
                    '0.2' : stellarModel(0.2,'Baraffe2015'),
                    '0.3' : stellarModel(0.3,'Baraffe2015'),
                    '0.4' : stellarModel(0.4,'Baraffe2015'),
                    '0.5' : stellarModel(0.5,'Baraffe2015'),
                    '0.6' : stellarModel(0.6,'Baraffe2015'),
                    '0.7' : stellarModel(0.7,'Baraffe2015'),
                    '0.8' : stellarModel(0.8,'Baraffe2015'),
                    '0.9' : stellarModel(0.9,'Baraffe2015'),
                    '1.0' : stellarModel(1.0,'Baraffe2015'),
                    '1.1' : stellarModel(1.1,'Baraffe2015'),
                    '1.2' : stellarModel(1.2,'Baraffe2015'),
                    '1.3' : stellarModel(1.3,'Baraffe2015'),
                    '1.4' : stellarModel(1.4,'Baraffe2015')}

'''==========================spin evolution setup============================'''
# =============================setting up torque laws===========================
def calculate_torque(omega_N,time_N,torque_law,X,interp_method,tc,tcX,tcY):
    '''called from spin evolution function. Redirects code to relevant torque
    # routine based on the keyword.
    In:
      omega_N: nth angular velocity [OmegaSun]
      time_N: nth time [Myr]
      torque_law: choice of torque law ('Matt2015','Matt2020' or 'Kawaler1998')
      X: LIST of torque parameters, to change torque parameters.
         Usually chi or ps, ks, p and beta.
      tc: Option to choose Amard2019 or Cranmer+Saar2011 turnover times.
          Note: if tcX and tcY != [], this option will be ignored
      tcX: list of knot point effective temperatures in K
      tcY: list of knot point effective temperatures in units of tau_sun
          (note: tcX and tcY are optional. When they are the default [],
          convective turnover time is instead calculated with Cranmer Saar
          2011 function)
    Other:
      interp_method: method of numberical integration, typically set to be RK2
    Out:
      age_array: final array of times [Myr]
      omega_array: final array of angular velocities [OmegaSun]'''

    if torque_law == 'Matt2015':
        T = calc_Matt2015_torque(omega_N,time_N,X,interp_method)

    elif (torque_law == 'Matt202X') or (torque_law == 'Matt2020'):
        T = calc_Matt202X_torque(omega_N,time_N,X,interp_method,tc,tcX,tcY)

    elif torque_law == 'Kawaler1998':
        T = calc_Kawaler1988_torque(omega_N,time_N,X,interp_method)

    return T
def calc_Matt2015_torque(omega_N,time_N,X,interp_method):
    ''' Calculates Matt 2015 torque for saturated and unsaturated regimes.
    in:
      omega_N = nth angular mom [const.OmegaSun]
    other:
      time_N:  age of star [Myr]
      X: list of torque parameters (usually only ever [chi])
      interp_method: method of interpolation to get stellar parameters
    out:
      T = torque [ergs]
    Notes:
      T0 and tauStar result in slightly different evolution to the simple
      torque: For Matt 2015 torque to reduce to the simple analytic form,
      change T0 and tauStar:
      T0 = (2./3.)*(9.5*10**30)*(1.**3.1)*(1.**0.5)
      tauStar = 1.
      For fixed critical Rossby Chi, have changing Omega_Crit.
      Chi = Omega_Crit*tauStar
      Chi = 10.
      Also, another way of determining saturated vs unsaturated regime is:
      if omega_N*tauStar > Chi:
          # Saturated
          T = -T0*(Chi**p)*omega_N
      else:
          # Unsaturated
          T = -T0*(tauStar**p)*(omega_N)**(p+1) '''

    # get stellar parameters at time_N:
    Rstar = SO.param_at_time('radiusStar',time_N,interp_method)[0]
    Mstar = SO.massStar
    tauStar = SO.param_at_time('turnoverTime',time_N,interp_method)[0]
    # chi = X[0]
    chi = const.CHI

    # calculate the torque
    p = 2
    T0 = (2./3.)*(9.5*10**30)*(Rstar**3.1)*(Mstar**0.5)  #With factor of 2/3!

    omegaCrit = chi/tauStar

    if omega_N > omegaCrit:
        T = -T0*(chi**p)*omega_N                #saturated regime
    else:
        T = -T0*(tauStar**p)*(omega_N)**(p+1)   #unsaturated regime

    return T
def calc_Matt202X_torque(omega_N,time_N,X,interp_method,tc,tcX,tcY):
    '''Calculates Matt 202X Torque.
    in:
      omega_N: nth angular mom [const.OmegaSun]
      time_N:  age of star [Myr]
      X: list of torque parameters [ps,ks,p,beta]
    other:
      interp_method: method of interpolation to get stellar parameters
      tc: turnover time prescription. Defaults to Cranmer and Saar 2011
          but can also implement Louis'
      tcX (optional): list of knot point effective temperatures in K
      tcY (optional): list of knot point effective temperatures in units of
                      tau_sun (note: tcX and tcY are optional. When they are
                      the default [], convective turnover time is instead
                      calculated with Cranmer Saar 2011 function)
    out:
      T = torque [ergs]
    Notes:
      To reduce Matt 202X torque to Matt 2015, set parameters to:
      ks = 100.
      ps = 0
      beta = 1.
      p = 2.
      ps = 0.35
      ks = 1000. '''

    ps = X[0]
    ks = X[1]
    p  = X[2]

    k1 = 6.20
    k2 = 0.0716


    # ===== get stellar parameters at time N =========
    Rstar = SO.param_at_time('radiusStar',time_N,interp_method)[0]
    Mstar = SO.massStar
    Teff = SO.param_at_time('Teff',time_N,interp_method)[0]

    # =====Turnover time==============================
    if len(tcX) > 1:
        # The following block of code is for when knot points only go up to a
        # certain Teff and then reduce to Cranmer+Saar 2011 at larger Teff:
        if SO.endknot != []:
            if Teff > SO.endknot[0]:
                # outside range of knot points, reduce to cramner and saar
                TauStar = SO.param_at_time('turnoverTime',time_N,\
                          interp_method)[0]
            else:
                # otherwise interpolate
                TauStar = SO.interpolatedTurnover(Teff)
        # The following block of code is for the entire Teff sequence
        # represented by knots
        else:
            TauStar = SO.interpolatedTurnover(Teff)

    # Or, the turnover time is just set by either Cranmer and Saar or Amard 2019
    elif tc == 'CranmerSaar':
        # Cranmer and Saar's turnover time
        TauStar = SO.param_at_time('turnoverTime',time_N,interp_method)[0]
    elif tc == 'Amard2019':
        # Louis' convective turnover time
        TauStar = louis_turnover_time(Teff)

    # ===== calculate the torque

    f = omega_N*const.OmegaSun*((Rstar*const.RSun)**3\
        /(const.GravConst*Mstar*const.MSun))**0.5

    if len(X) == 4:
        beta = X[3]
    else:
        beta = (1.+(f/k2)**2)**0.5

    RossbyStar = (omega_N*TauStar)**-1   #[OmegaSun*TauSun]

    Fm = min(ks*(RossbyStar)**ps,(RossbyStar)**(-p))

    T0 = -6.33333333333e30*2./p

    T = T0*(Rstar**3.1)*(Mstar**0.5)*omega_N*(beta**(-0.44))*Fm

    return T
def calc_Kawaler1988_torque(omega_env,time_N,X,interp_method):
    '''Calculates torque from Kawaler 1998. Values from Spada2011
    in:
      omega_env = nth angular mom of envelope [const.OmegaSun]
      time_N:  age of star [Myr]
      X: list of torque parameters (never been used with this torque)
    other:
      interp_method: method of interpolation to get stellar parameters
      constant values from Kawaler1998
      Kw = 5.97e47
      omegaCrit = 5.5 OmegaSun
    out:
      T = torque [ergs]'''

    RStar = SO.param_at_time('radiusStar',time_N,interp_method)[0]
    MStar = SO.massStar
    Kw = 5.97e47
    omegaSat = 5.5

    T = -Kw*(RStar/MStar)**0.5\
        *min((omega_env*const.OmegaSun)**3,\
        (omegaSat*const.OmegaSun)**2*omega_env*const.OmegaSun)

    return T

# ===========================setting up turnover times=========================
def define_knots_entireSequence():
    # defines 14 knot points in turnover time vs effective temperature
    # function (i.e. this function is represented as a series of
    # x and y coordinates, where x is T_eff and y tau_c. Values
    # between knot points are linearly interpolated in the models.
    # The y coordinates are randomly varied for an optimal fit
    # to observed data using MCMC.)
    # out:
    #   tcX: x knots array of effective temperatures [K]
    #   tcY: y knots array of turnover times [Tau_c,sun]

    '''======== x knots =========='''
    # defining the extremes
    tcX = [2808]

    # what is Teff for each mass at age of Praesepe?
    for i in range(len(const.massArray)):
        global SO
        if i != 0:
            SO = stellarDictionary['%s' %const.massArray[i]]
            tcX.append(SO.param_at_time('Teff',665.)[0])
    tcX.append(7000)

    '''========= y knots ========='''
    # find corresponding Cranmer and Saar y knots (units of tau_sun)
    tcY=[]
    for i in range(len(tcX)):
        tcY.append((314.24*np.exp(-(tcX[i]/1952.5)-(tcX[i]/6250.)**18)+0.002)\
        /const.TauSun)

    return tcX, tcY
def interpolate_tauc(tcX,tcY,endknot):
    # fits a function to knot points defined by Teff and tauc
    # coordinates tcX and tcY respectively. This function allows turnover time
    # to either be fully expressed as a linear interpolation of tcX and tcY, or
    # a combination if endknot is defined. If endknot is given, the turnover
    # time reverts to Cranmer and Saar 2011 above a certain Teff.
    # in:
    #   tcX: x knots array of effective temperatures [K]
    #   tcY: y knots array of turnover times [Tau_c,sun]
    #   endknot: boundary defining switch from knot points to function [K]
    temp_x = []
    temp_y = []

    for i in range(len(tcX)):
        temp_x.append(tcX[i])
        temp_y.append(tcY[i])

    if endknot != []:
        temp_x.append(endknot[0])
        temp_y.append(endknot[1])

    linearFit = scipy.interpolate.interp1d(temp_x,temp_y,kind='linear')

    return linearFit
def louis_turnover_time(Teff):
    # returns Louis Amard's convective turnover time in tauSun
    # in:
    #   Teff: effective temperature [K]
    # out:
    #   tau: turnover time [tau_c,sun]
    if (Teff<=3480.):
        tauLouis = 10.**(6.52112823e-7*Teff**2. - 4.00355099e-3*Teff
                    + 8.68234621)
    else:
        tauLouis = 10.**(-2.51904051e-10*Teff**3. + 3.73613409e-6*Teff**2.
                    - 1.85566042e-2*Teff + 32.5950535)

    #renormalise to Cranmer & Saar 2011 scale
    tau = tauLouis*13.79/35.54/const.TauSun

    return tau

# ====================calculating spin evolution for one star===================
def calc_solid_body_spin_evol(omega_N,time_N,time_F=[],
                              torque_law='Matt2015',X=[const.CHI],
                              tcX=[],tcY=[],tc='CranmerSaar',
                              interp_method='linearSpline',
                              tolerance=const.TOL):
    '''Calculates spin evolution for one star.
    In:
      omega_N: nth angular velocity, which is the initial condition [OmegaSun]
      time_N: nth time, which is the initial time condition [Myr]
      time_F: final time after which the evolution is no longer calculated [Myr]
      torque_law: choice of torque law ('Matt2015','Matt2020' or 'Kawaler1998')
      X: LIST of torque parameters, to change torque parameters.
         Usually chi or ps, ks, p and beta.
      tc: Option to choose Amard2019 or Cranmer+Saar2011 turnover times.
          Note: if tcX and tcY != [], this option will be ignored
      tcX: list of knot point effective temperatures in K
      tcY: list of knot point effective temperatures in units of tau_sun
          (note: tcX and tcY are optional. When they are the default [],
          convective turnover time is instead calculated with Cranmer Saar
          2011 function)
      tolerance: for varying timestep, from a constants file by default [%]
    Other:
      IntegrationMethod: is set to be RK2
      timestep: is set to be variable
      omega_N: angular velocity is updated on each iteration of loop [OmegaSun]
      time_N: time used on current iteration of loop [Myr]
    Out:
      age_array: final array of times [Myr]
      omega_array: final array of angular velocities [OmegaSun]'''

    #========set up initial conditions========================
    omega_array = [omega_N]
    age_array = [time_N]
    # calculate spin evolution
    time_F = 0.8*SO.time[-1]

    #Loop through time until specified final time
    while time_N <= time_F:

    # =================================calculate T/I============================
        I = SO.param_at_time('inertiaStar',time_N,interp_method)
        SO.index = I[1]
        I = I[0]

        dIdt = SO.param_at_time('dIdtStar',time_N,interp_method)[0]

        k1 = calculate_torque(omega_N,time_N,torque_law,X,
                              interp_method,tc,tcX,tcY)/I

        # ==============Calculate new, variable timestep [s]:===================
        # choose the smallest term in angular momentum equation to calculate dt
        if dIdt != 0.0 and abs(k1) != 0.0:
            f1  = 1./k1
            f2 = I/(omega_N*const.OmegaSun)/dIdt
            if  abs(f2) < abs(f1):
                dt = omega_N*const.OmegaSun*tolerance*abs(f2)
            else:
                dt = omega_N*const.OmegaSun*tolerance*abs(f1)
        # for crazy torque parameters, k1 is zero (no torque) and f1 is inf.
        # First condition above calculates dt using f2 but gives an error
        # message, which I want to avoid.
        elif abs(k1) == 0.0 and dIdt != 0.0:
            f2 = I/(omega_N*const.OmegaSun)/dIdt
            dt = omega_N*const.OmegaSun*tolerance*abs(f2)
        # when there is no change in stellar structure (dIdt=0), f2 becomes inf.
        # First condition calculates dt using f1 but gives an error message,
        # which I want to avoid.
        elif dIdt == 0.0 and abs(k1) != 0.0:
            f1  = 1./k1
            dt = omega_N*const.OmegaSun*tolerance*abs(f1)
        # for crazy torque parameters at time of stable structure,
        #  both terms can be 0.0.
        # In this case, no rotational evolution is expected to occur.
        elif dIdt == 0.0 and abs(k1) == 0.0:
            dt = 1000.*const.Myr
        else:
            sys.exit('sys.exit ERROR: No timestep condition was fulfilled.')

        # =================calculate next step of evolution=====================
        k2 = calculate_torque(omega_N + k1*dt/\
             const.OmegaSun,time_N,torque_law,X,interp_method,tc,tcX,tcY)/I
        omega_N = (0.5*k1 + 0.5*k2)*dt/const.OmegaSun -\
                  (omega_N*dIdt*dt)/I + omega_N

        # ====================update and save===================================
        time_N = time_N + dt/const.Myr
        #Fill arrays with new time and angular velocity
        age_array.append(time_N)
        omega_array.append(omega_N)

    # reset global value to 0 for next star to be calculated
    SO.index=0

    return age_array,omega_array

'''===================brute force probability densities======================'''
# how to use: switch options in myconstants.py for range of masses and initial
# rotation rates. calc_cluster_sb_evol will call
# calc_solid_body_spin_evol and return a stellar dictionary of the
# spin tracks. The PMD can then be generated from the dictionary using
# calculate_PMD.
# This can be further binned into probability densities and a tau squared
# calculated.
# example code:
# $ calc_cluster_sb_evol(5.,1000)
# $ PMD = calculate_PMD(665.)
# $ linMatrix = make_linear_matrix(PMD[0],PMD[1])[0]
# $ tauSquared = tau2.calculateTauSquared(\
#                Praesepe.masses,Praesepe.periods,linMatrix,ymin,ymax,modelProb)

def calc_cluster_sb_evol(time_N,time_F,
                                    torque_law='Matt2015',X=[const.CHI],tc=[],
                                    tcX=[],tcY=[],endknot=[],
                                    CTOTParameters=[],
                                    interp_method='linearSpline',
                                    tolerance=const.TOL):
    ''' calculates rotational evolution for a cluster of stars (brute force)
    In:
      time_N: nth time, which is the initial time condition [Myr]
      time_F: final time after which the evolution is no longer calculated [Myr]
      torque_law: choice of torque law ('Matt2015','Matt2020' or 'Kawaler1998')
      X: LIST of torque parameters, to change torque parameters.
         Usually chi or ps, ks, p and beta.
      tc: Option to choose Amard2019 or Cranmer+Saar2011 turnover times.
          Note: if tcX and tcY != [], this option will be ignored
      tcX: list of knot point effective temperatures in K
      tcY: list of knot point effective temperatures in units of tau_sun
          (note: tcX and tcY are optional. When they are the default [],
          convective turnover time is instead calculated with Cranmer Saar
          2011 function)
      tolerance: for varying timestep, from a constants file by default [%]
    Out:
      updated stellarDictionary containing spin tracks of each star'''

    # following lines of code only for using knot points in conv turnover time
    if tcX !=[]:
        interpFunction = interpolate_tauc(tcX,tcY,endknot)
    else: interpFunction = []

    # cycling through each mass
    for i in range(len(const.massArray)):
        # setting up each mass
        global SO

        SO = stellarDictionary['%s' %const.massArray[i]]
        print('mass:',const.massArray[i])
        SO.endknot = endknot
        SO.interpolatedTurnover = interpFunction

        # for new run, make sure that no old spin evol data is stored
        stellarDictionary['%s' %const.massArray[i]].spinEvolOmega = []
        stellarDictionary['%s' %const.massArray[i]].spinEvolTime = []

        # cycling through initial rotation rates
        for j in range(const.NInitialConditions):

            spinEvol = calc_solid_body_spin_evol(const.omegaN[j],time_N,time_F,
                                                        torque_law,X,
                                                        tcX,tcY,tc,
                                                        interp_method,
                                                        tolerance)

            stellarDictionary['%s' %const.massArray[i]].spinEvolOmega.append\
                                                                (spinEvol[1])
            stellarDictionary['%s' %const.massArray[i]].spinEvolTime.append\
                                                                (spinEvol[0])

    return stellarDictionary
def calculate_PMD(age,tcX=[],turnoverParametersY=[]):

    # interpolates periods (and log(period)s) and masses from calculated cluster
    # spin evolution, which is stored in the stellar dictionary. This can then
    # be passed to make_linear_matrix or make_log_matrix to construct
    # probability densities.
    # in:
    #   age: an input age at which the PMD is calculated [Myr]
    # out:
    #   flatlistMasses: a list of all masses in the cluster
    #   flatlistPeriods: a corresponding list of the rotation periods in cluster
    #   flatlistPeriods: a corresponding list of log(rotation periods)

    periods = []
    masses = []
    logP = []
    for i in range(len(const.massArray)):
        periods.append(stellarDictionary['%s' %const.massArray[i]].\
                interpolateSpinEvolution(age,\
                const.massArray[i],turnoverParametersX,turnoverParametersY))
        logP.append(np.log10(stellarDictionary['%s' %const.massArray[i]].\
                interpolateSpinEvolution(age,\
                const.massArray[i],turnoverParametersX,turnoverParametersY)))
        masses.append(np.full(const.NInitialConditions,const.massArray[i]))

    # convert to big list instead of lists of lists
    flatlistPeriods = [item for sublist in periods for item in sublist]
    flatlistMasses = [item for sublist in masses for item in sublist]
    flatlistLogP = [item for sublist in logP for item in sublist]
    # check if array has nans
    if np.isnan(np.sum(flatlistPeriods)) == True:
        sys.exit('ERROR, PMD has found NAN values: %s' \
                 %np.isnan(np.sum(flatlistPeriods)) )

    return flatlistMasses, flatlistPeriods,flatlistLogP
def calc_cell_area_log(k=0):
    # Calculates the linear area of each bin (log) using global variables.
    # In the log case, the area is not constant in linear space due to the
    # even binning in log space. In this case, an integer k corresponding to
    # the vertical position is required.
    # Assumes a matrix is laid out as the following:
    # (most convenient for plotting):
    #   |k=0|k=0|       k=0
    #   |k=1|k=1|       k=1
    #   |k=2|k=2|       k=2
    #
    # in:
    #   k: integer corresponding to the vertical position of desired bin
    # requires(global):
    #   ymax: MUST be the log of linear ymax
    #   ymin: MUST be the log of linear ymin
    #   xmax: max x limit
    #   xmin: min x limit
    #   N: Number of bins in x
    #   M: Number of bins in y
    # out:
    #   area: LINEAR area of desired bin whose boundaries are defined in
    #           log [Msun*days]

    # bin spacing in log space (constant)
    logHeight = (const.logYmax - const.logYmin)/float(const.M)
    # bin spacing in linear space (varies from bin to bin)
    linearUpperBoundary = 10**(const.logYmax - k*logHeight)
    linearLowerBoundary = 10**(const.logYmax - (k+1)*logHeight)

    # area in linear space
    height = linearUpperBoundary - linearLowerBoundary
    width = (const.xmax-const.xmin)/float(const.N)
    area = height*width

    return area
def make_linear_matrix(x_array,y_array,initialConditions='N/A'):
    # Makes MxN number density matrix (rho) of x and y model data.
    # Requires xmin, xmax, ymin, ymax to be limits in linear space.
    # Output matrix has layout (most convenient for plotting):
    # |matrix[0,0]|matrix[0,1]| i = 0
    # |matrix[1,0]|matrix[1,1]| i = 1
    #   j=0   j=1
    # Be careful, y is the first integer of the matrix and x the second.
    # in:
    #   x_array: array of x data, len(x_array) [float]
    #   y_array: array of y data, len(x_array) [float]
    # out:
    #   matrix: MxN matrix that has been normalised to total number of stars
    #           and area of grid cell [(Msun*days)^-1]
    #   numDensityMatrix: Number density MxN matrix, normalised to the total
    #           number of model stars [unitless]
    allPeriods = []
    allProbs = []
    for i in range(len(const.massArray)):
        f = open('%sMSun_KDE.txt' %const.massArray[i])
        lines = f.readlines()[1:]
        f.close()

        periods = []
        logProb = []

        #Filling lists for desired mass
        for line in lines:
            data = line.split()
            periods.append(float(data[0]))
            logProb.append(float(data[1]))
            allPeriods.append(float(data[0]))
            allProbs.append(float(data[1]))


    n_stars = len(x_array)

    # Creating an empty matrix
    matrix = [[0.0 for x in range(const.N)] for y in range(const.M)]
    numDensityMatrix = [[0.0 for x in range(const.N)] for y in range(const.M)]

    # FILLING matrix bins with model points using global variables
    for i in range(len(x_array)):
        # for crazy initial conditions and torques, some spin tracks will not be
        #  contained within grid.
        # Therefore set the maximum and minimum period values to the extremities
        #  of the grid.
        if y_array[i] > const.ymax:
            y_array[i] = const.ymax
        if y_array[i] < const.ymin:
            y_array[i] = const.ymin
        # find the appropriate x and y bin for model point to fall into
        y_bin = int((y_array[i] - const.ymin)*const.M/(const.ymax - const.ymin))
        x_bin = int((x_array[i] - const.xmin)*const.N/(const.xmax - const.xmin))

        # some model points are weighted by initial conditions

        if initialConditions == 'Matt2020':
            # matt 2020 tophat imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin] +\
                                             const.Matt2020Probabilities[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] =\
                                     numDensityMatrix[const.M-1-y_bin][x_bin]\
                                         + const.Matt2020Probabilities[i]
        elif initialConditions == 'Matt2015':
            # matt 2015 tophat imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin] +\
                                             const.Matt2015Probabilities[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] =\
                                    numDensityMatrix[const.M-1-y_bin][x_bin]\
                                    + const.Matt2015Probabilities[i]
        elif initialConditions == 'UpperSco':
            # taking upper sco stellar weighting into account

            # finding the relevant probability
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin] +\
                                             allProbs[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] =\
                                    numDensityMatrix[const.M-1-y_bin][x_bin]\
                                    + allProbs[i]

        else:
            # no function imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] += 1
            numDensityMatrix[const.M-1-y_bin][x_bin] += 1

    # Normalising matrix by gridcell area and total number of stars
    totDensity = 0
    for i in range(const.M): #period loop
        for j in range(const.N):#mass loop
            if initialConditions == 'N/A':
                matrix[i][j] = float(matrix[i][j])/float(const.linearPixelArea)\
                              /float(n_stars)
                numDensityMatrix[i][j] = float(numDensityMatrix[i][j])\
                                        /float(n_stars)
            else:
                # The probabilities associated with the initial conditions sum
                # to 1 in each mass bin.
                # Just need to renormalise by mass and area for the number
                # density and area density matrices
                matrix[i][j] = float(matrix[i][j])/len(const.massArray)/\
                               float(const.linearPixelArea)

                numDensityMatrix[i][j] = float(numDensityMatrix[i][j])/\
                               len(const.massArray)
            totDensity = numDensityMatrix[i][j] + totDensity

    return matrix,numDensityMatrix
def make_log_matrix(x_array,y_array,stellarWeighting='N/A',
                 initialConditions='N/A'):
    # Makes MxN number density matrix (rho) of x and y observed data.
    # Requires xmin, xmax, ymin, ymax to be limits in linear space.
    # Output matrix has layout (most convenient for plotting):
    # |matrix[0,0]|matrix[0,1]| i = 0
    # |matrix[1,0]|matrix[1,1]| i = 1
    #   j=0   j=1
    # Be careful, y is the first integer of the matrix and x the second.
    # in:
    #   x_array: x data [float]
    #   y_array: y data [float]
    # out:
    #   matrix: MxN matrix that has been normalised to total number of stars and
    #       area of grid cell [(Msun*days)^-1]
    #   pixelArea: MxN matrix corresponding to above matrix, where each pixel
    #       corresponds to the area of a pixel in matrix
    n_stars = len(x_array)

    # Creating an empty matrix
    matrix = [[0.0 for x in range(const.N)] for y in range(const.M)]
    pixelArea = [[0.0 for x in range(const.N)] for y in range(const.M)]
    numDensityMatrix = [[0.0 for x in range(const.N)] for y in range(const.M)]
    # FILLING matrix bins using global variables
    for i in range(len(x_array)):
        # for crazy initial conditions and torques, some spin tracks will not be
        #  contained within grid.
        # Therefore set the maximum and minimum period values to the extremities
        #  of the grid.
        if y_array[i] > const.logYmax:
            y_array[i] = const.logYmax
        if y_array[i] < const.logYmin:
            y_array[i] = const.logYmin
        # find the appropriate x and y bin for model point to fall into
        y_bin = int((y_array[i] - const.logYmin)*const.M\
                /(const.logYmax - const.logYmin))
        x_bin = int((x_array[i] - const.xmin)*const.N\
                /(const.xmax - const.xmin))
        # some model points are weighted by initial conditions
        if initialConditions == 'Matt2020':
            # matt 2020 tophat imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin]\
                                            + const.Matt2020Probabilities[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] =\
                                    numDensityMatrix[const.M-1-y_bin][x_bin]\
                                     + const.Matt2020Probabilities[i]

        elif initialConditions == 'Matt2015':
            # matt 2015 tophat imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin]\
                                            + const.Matt2015Probabilities[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] = \
                                    numDensityMatrix[const.M-1-y_bin][x_bin]\
                                    + const.Matt2015Probabilities[i]

        elif initialConditions == 'UpperSco':
            # taking upper sco stellar weighting into account
            matrix[const.M-1-y_bin][x_bin] = matrix[const.M-1-y_bin][x_bin]\
                                            + const.UpperScoProbabilities[i]
            numDensityMatrix[const.M-1-y_bin][x_bin] = \
                                    numDensityMatrix[const.M-1-y_bin][x_bin]\
                                     + const.UpperScoProbabilities[i]
        else:
            # no function imposed on initial conditions
            matrix[const.M-1-y_bin][x_bin] += 1
            numDensityMatrix[const.M-1-y_bin][x_bin] += 1

    totDensity = 0
    # Make normalised matrix
    for y in range(const.M):          #Loop over y axis
        for x in range(const.N):      #Loop over x axis
            # Next calculate the normalised density of a grid cell
            pixelArea[y][x] = calc_cell_area_log(y)
            if initialConditions == 'N/A':

                # no function imposed on initial conditions
                matrix[y][x] = float(matrix[y][x])/float(pixelArea[y][x])\
                              /float(n_stars)
                numDensityMatrix[y][x] = float(numDensityMatrix[y][x])\
                              /float(n_stars)
            else:
                # The probabilities associated with the initial conditions
                # sum to 1 in each mass bin.
                # Just need to renormalise by mass and area for the number
                # density and area density matrices
                matrix[y][x] = float(matrix[y][x])/len(const.massArray)\
                              /float(pixelArea[y][x])
                numDensityMatrix[y][x] = float(numDensityMatrix[y][x])\
                              /len(const.massArray)
            totDensity = numDensityMatrix[y][x] + totDensity


    return matrix,pixelArea,numDensityMatrix

'''===============Deterministic method probability densities================='''
def generate_PiPf_spacing(age,time_N,time_F,
                                    torque_law,X,
                                    tcX,tcY,N,
                                    tol=0.2,mass=1.0):

    # generates spacing of Pi(Pf) curves which are needed to construct
    # probability densities in the deterministic method.
    # in:
    #   age: desired age of eventual PMD [Myr]
    #   timeN: the initial age from which to set the spin evolution [Myr]
    #   timeF: the final age of the spin evolution [Myr]
    #   torque_law: the torque with which to evolve the spin (i.e. "Matt2020")
    #   X: LIST of torque parameters, to change torque parameters.
    #      Usually chi or ps, ks, p and beta.
    #   tcX: list of knot point effective temperatures in K
    #   tcY: list of knot point effective temperatures in units of tau_sun
    #   N: number of points used to define Pi(Pf) curves
    #   tol: tolerance for spin evolution tracks
    #   mass: mass for which to construct the Pi(Pf) curve
    # out:
    #   y_n: the final period Pf (days)
    #   x_n: the initial period Pi (days)

    # construct the initial spacing of Pi
    initialPeriods = [0.5]
    initialPeriods.extend(np.linspace(0.7,18.,N))
    initialPeriods.extend([19.])
    initialPeriods = np.array(initialPeriods)
    initialOmegas = 2*math.pi /(initialPeriods*const.day)/const.OmegaSun
    x_n = []
    y_n = []

    # for each Pi, find the corresponding Pf by evolving the rotation rate
    for i in range(len(initialOmegas)):
        # evolve the rotation rate
        SB = calc_solid_body_spin_evol(initialOmegas[i],time_N,time_F,
                                            torque_law,X,
                                            tcX,tcY)
        # find Pf at desired age
        ageInterpolation = interp1d(np.array(SB[0]),np.array(SB[1]),\
                           kind='quadratic')
        omega_f = ageInterpolation(age)

        Pi = 2*math.pi / (initialOmegas[i]*const.OmegaSun) / const.day
        Pf = 2*math.pi / (omega_f*const.OmegaSun) / const.day

        if i == 0:
            x_n.append(Pi)  #in days
            y_n.append(Pf)  #in days
        if (i > 0) and (Pf > y_n[-1]):
            x_n.append(Pi)  #in days
            y_n.append(Pf)  #in days

    return y_n,x_n
def generate_splines(mass,age,time_N,time_F,torque_law,X,tcX,tcY,
                    endknot,tol,N):

        # constructs the splines needed to make probability densities. Updates
        # the stellarDictionary.

        # ====== generate Pi(Pf) curves ======
        PiPf = generate_PiPf_spacing(age,time_N,time_F,
                                torque_law,X,
                                tcX,tcY,N,tol)

        SO.Pf = np.array(PiPf[0])
        SO.Pi = np.array(PiPf[1])
        SO.N = N

        # interpolate Pi(Pf)
        SO.PiPfInterpolated = interp1d(SO.Pf,SO.Pi,kind='quadratic',
                              bounds_error=False,fill_value=(0,0))
        SO.PfPiInterpolated = interp1d(SO.Pi,SO.Pf,kind='quadratic',
                              bounds_error=False,fill_value=(0,0))

        # fit a spline to Pi(Pf)
        SO.PfPi_univariateSpline = scipy.interpolate.UnivariateSpline(
                                   SO.Pf,SO.Pi,s=0)
        SO.PiPf_univariateSpline = scipy.interpolate.UnivariateSpline(
                                   SO.Pi,SO.Pf,s=0)

        # ====== calculate dPi/dPf ==========
        # commented out because it's easier to construct elsewhere - but this is
        # the method:
        #
        # use the lines of code:
        #   a = np.linspace(min(SO.Pf),max(SO.Pf),100000)
        #   splin.append(SO.PfPi_univariateSpline.derivatives(a[k])[1])
        # other (less efficient ways) of generating dPi/dPf:
        # # use np.gradient with reference stars
        # # to plot dPi_dPf, plot against SO.Pf
        # stellarDictionary['%s' %mass].dPi_dPf =\
        #                                  np.gradient(SO.Pi,SO.Pf,edge_order=1)
        #
        # # using the interpolated Pi Pf quadratic
        # a =np.linspace(min(SO.Pf),max(SO.Pf),1000)
        # indices = np.where(np.array(PiPf[2](a)) != 0.0)[0]
        # a = a[indices]
        #
        # gradInterpolated = np.gradient(PiPf[2](a),a,edge_order=1)
        # stellarDictionary['%s' %mass].gradInterpolated = (a,gradInterpolated)

        # ======= things needed to calculate rho ========
        SO.Matt2020norm = 1./np.log(18./0.7)/1.3
        SO.edgePf1Matt2020 = SO.PiPf_univariateSpline.__call__(0.7)
        SO.edgePf2Matt2020 = SO.PiPf_univariateSpline.__call__(18.)

def generate_cluster_prob_density(age=665.,time_N=5.,
                                  time_F=6000.,torque_law='Matt2015',
                                  X=[const.CHI],tcX=[],tcY=[],endknot=[],
                                  tol=0.2,N=20):
    # generates spline fits to Pi(Pf curves) and stores constants for
    # construction of rho(Pf) curves.
    # in:
    #   age: desired age of synthetic cluster [Myr]
    #   timeN: final time of spin Evolution [Myr]
    #          (for our purposes, initial spin evolution time is always assumed
    #          to be 5Myr)
    #   torque_law: torque law used to evolve spin rates (e.g. "Matt2020")
    # X: LIST of torque parameters, to change torque parameters.
    #       Usually chi or ps, ks, p and beta.
    # tcX: list of knot point effective temperatures in K
    # tcY: list of knot point effective temperatures in units of tau_sun
    #   (note: tcX and tcY are optional. When they are the default [],
    #   convective turnover time is instead calculated with Cranmer Saar
    #   2011 function)
    #  endknot: boundary defining switch from knot points to function [K]
    #   N: number of points used to define Pi(Pf) curves
    #   tol: tolerance for spin evolution tracks
    # out:
    #   updated stellarDictionary

    # first of all, decide how convective turnover times should be defined.
    interpFunction = interpolate_tauc(tcX,tcY,endknot=endknot)
    for i in range(len(const.massArray)):

        global SO
        SO = stellarDictionary['%s' %const.massArray[i]]

        SO.endknot = endknot
        SO.interpolatedTurnover = interpFunction
        generate_splines(const.massArray[i],age,time_N,time_F,
                        torque_law,X,tcX,tcY,endknot,tol,N)

    return stellarDictionary
