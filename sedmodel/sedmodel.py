"""
Implementation of the superposition model for
calculating gas flow in straight channels.
For details, see "Determining Molecular Diameters of Rarefied Gases".

All units are SI units.
"""

import numpy as np
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

#import helpers as hlp
from eznukutils import (
    arrayutils, pythonutils as putils, rgd, physics as ph)


class sed_base(ABC):
    
    # ----------------------------------------------
    # to be implemented by geometry-specific child class:
    
    @abstractmethod
    def get_mdot_slip(self):
        """Convective part of the flow including the slip influence."""
        ...
    
    @abstractmethod
    def _get_smol_A(self):
        """
        Gometric-dependent part of the FM expression
        by Smoluchowski.
        When using the numerical approach, use something like:

        if not hasattr(self, "_smol_A"):
            import smoluchowski
            from shapely.geometry import Polygon
            # define the cross section here; a rectangular cross
            # section would look like this:
            p1 = [0, 0]
            p2 = [0, self.h]
            p3 = [self.w, self.h]
            p4 = [self.w, 0]
            area = Polygon(shell=(p1, p2, p3, p4))
            self._smol_A = smoluchowski.get_smol_int(area)
        return self._smol_A
        """
        ...
        
    # ----------------------------------------------
        
    def set_params(self, params):
        """
        Sets the parameters of the model.

        Parameters
        ----------
        params : dict
            Dictionary consisting the model parameters.

        Returns
        -------
        None.

        """
        self.Kn = arrayutils.convert_to_array(params["Kn"])
        self.gas = params["gas"]
        self.T, self.p_r = putils.get_opt_kwargs(params, T=293.15, p_r=10)
        
        # optional keyword parameters:
        self.sigma, self.d_gas, self.L = putils.get_opt_kwargs(params,
                                                               sigma=1,
                                                               d_gas=None,
                                                               L=1)
        
        
    def calc_G(self):
        """
        calculates the dimensionless mass flow.
        """
        self.calc_mdot()
        self.G = rgd.mdot_to_g(self.mdot, self.L, 
                               self.P, self.A, 
                               self.dp, self.T, self.gas)
            
    def fit_fun(self, X, *fit_vals):
        """
        function used by fit_to_data
        """
        Kn_data, fit_keys = X.values()
        # update the attributes
        for fit_key, fit_val in zip(fit_keys, fit_vals):
            setattr(self, fit_key, fit_val)
        self.calc_G()
        return self.G
    
    def fit_to_data(self, Kn_data, G_data, update=True, 
                    bounds=(-np.inf, np.inf), **to_fit):
        """
        Fits the model to data.

        Parameters
        ----------
        Kn_data : array_like
            Knudsen data points.
        G_data : array_like
            Dimensionless mass flow data points.
        update : bool, optional
            Update the model parameters after fitting to the
            optimized values? The default is True.
        bounds : tuple, optional
            Bounds for the optimization. The default is (-np.inf, np.inf).
        **to_fit : float
            Keyword arguments to specify the parameters to fit and their
            initial guess, e.g. d_gas=300e-12.

        Returns
        -------
        tuple
            optimization results. First entry is the optimzed parameter(s).

        """
        
        # back up the actual values
        old_vals = [getattr(self, key) for key in to_fit.keys()]
        old_Kn = self.Kn
        
        self.Kn = Kn_data
        
        X = {"a": Kn_data,
             "b": list(to_fit.keys())}
        p0 = list(to_fit.values())
        self.popt, self.pcov = curve_fit(self.fit_fun, X, 
                                         G_data, p0=p0, bounds=bounds)
        
        if update:
            for key, opt_val in zip(to_fit.keys(), self.popt):
                setattr(self, key, opt_val)
        else:
            # setting back the old attributes
            for key, old_val in zip(to_fit.keys(), old_vals):
                setattr(self, key, old_val)
        
        self.Kn = old_Kn
        self.calc_G()
        
        return self.popt
            
    def get_pred_interv(self, alpha=0.05):
        """
        Calculates the prediction interval(s) of the fit.

        Parameters
        ----------
        alpha : float, optional
            Interval size. The default is 0.05.

        Raises
        ------
        ValueError
            if no fit is calculated yet.

        Returns
        -------
        intervals : list, float
            list of intervals for the optimized parameters.

        """
        
        if hasattr(self, "pcov"):
            # "Predictive Inference" by Seymour Geisser, p. 9
            # alpha = 0.05 means 95% two-sided confidence interval
            
            from scipy.stats.distributions import t
            
            n = len(self.Kn)    # number of data points
            p = len(self.popt)  # number of parameters
            
            dof = max(0, n - p)         # number of degrees of freedom
            
            # student-t value for the dof and confidence level
            tval = t.ppf(1-alpha/2, dof) 
            
            intervals = []
            for this_pcov in np.diag(self.pcov):
                sigma = this_pcov ** 0.5
                intervals.append(sigma*tval*np.sqrt(1+1/n))
                
            return intervals
        else:
            raise ValueError("no fit calculated yet")
    
    def get_mdot_self_diff(self):
        """
        Calculates the self diffusive mass flow.

        Returns
        -------
        mdot : array_like
            Mass flow due to self diffusion.

        """
        
        M = ph.get_M(self.gas)
        R = ph.R
        p_m = (self.pin+self.pout)/2
        u = np.sqrt(8*R*self.T/(np.pi*M))     # mean molecular velocity
        if self.d_gas:
            mfp = rgd.mfp(self.T,self.d_gas,p_m)
        else:
            mfp = rgd.mfp_visc(self.T, p_m, gas=self.gas)
            
        diff_coeff = 1/3 * mfp * u
        mdot = self.dp * diff_coeff * self.A/self.L * M/(R*self.T)
        return mdot
    
    
    def get_mdot_FM_smoluchowski(self):
        # general FM expression by Smoluchowski.
        # A_smol is the geometry-dependent part:
        smol_A = self._get_smol_A()
        alpha = (2-self.sigma)/self.sigma
        mdot_FM = (alpha * 1/(2*np.sqrt(2*np.pi)) 
                * np.sqrt(ph.get_M(self.gas)/(ph.R*self.T))
                * (self.pin-self.pout) / self.L * smol_A )
        return mdot_FM
    
    def calc_mdot(self, calc_pressures_from_Kn=True):
        """
        Calculates the mass flow for each Kn number.

        Returns
        -------
        None.

        """
        if calc_pressures_from_Kn:
            # The pressures are calculated with mfp from viscosity.
            self.pin, self.pout = rgd.get_pin_pout(self.Kn, self.D_H,
                                                   self.T, self.gas, 
                                                   d_gas=None,
                                                   ratio=self.p_r)

        self.dp = self.pin - self.pout
        self.mdot_slip = self.get_mdot_slip()
        self.mdot_self_diff = self.get_mdot_self_diff()
        self.mdot_FM = self.get_mdot_FM_smoluchowski()
        self.mdot_eff_diff = 1 / (1/self.mdot_self_diff + 1/self.mdot_FM)
        self.mdot = self.mdot_slip + self.mdot_eff_diff
    
        
class sed_circ(sed_base):
    def __init__(self, params=None):
        if params:
            self.set_params(params)
            
    def set_params(self, params):
        """
        Sets the parameters of the model.

        Parameters
        ----------
        params : dict
            Dictionary consisting the model parameters.

        Returns
        -------
        None.
        
        """
        
        super().set_params(params)
        self.D, = putils.get_opt_kwargs(params, D=1)
        self.D_H = self.D
        self.A = np.pi/4*self.D**2
        self.P = np.pi*self.D
        
    def _get_smol_A(self):
        # geometric-dependent part of the FM expression
        # by Smoluchowski, here for circular cross sections
        return 16 * (self.D/2)**3 * np.pi / 3
        
    def get_mdot_poiseuille(self):
        """
        Hagen Poiseuille mass flow according to Hadj-Nacer et al. (2014):
        Gas flow through microtubes with different internal surface coatings 
        """
        p_m = (self.pin+self.pout)/2
        mu = ph.get_mu(self.gas, self.T)
        M = ph.get_M(self.gas)
        mdot = np.pi*self.D**4*self.dp*p_m / (128*mu*ph.R/M*self.T*self.L)

        return mdot
        
    def get_slip_factor(self):
        """
        Calculates and returns the "slip factor", the factor by which
        the Hagen-Poiseuille flow is enhanced by slip.
        See manuscript for derivation.
        
        """
        if self.d_gas:
            mfp = rgd.mfp(self.T, self.d_gas, (self.pin+self.pout)/2)
        else:
            mfp = rgd.mfp_visc(self.T, (self.pin+self.pout)/2, self.gas)

        mfp_eff = 1/(1/mfp + 1/(self.D))
        alpha = (2 - self.sigma) / self.sigma
        S = 1 + 8 * alpha * mfp_eff / self.D_H
        return S
    
    
    def get_mdot_slip(self):
        return self.get_mdot_poiseuille() * self.get_slip_factor()
    
        
class sed_rect(sed_base):
    def __init__(self, params):
        if params:
            self.set_params(params)
        # this is the adjustment to the TMAC of the slip
        # expression as explained in the manuscript:
        self.sigma_slip = 0.9
            
    def set_params(self, params):
        """
        Sets the parameters of the model.

        Parameters
        ----------
        params : dict
            Dictionary consisting the model parameters.

        Returns
        -------
        None.
        
        """
        
        super().set_params(params)
        self.h = params["h"]
        self.w = params["w"]
        self.D_H = self.h
        self.A = self.h*self.w
        self.P = 2*(self.h+self.w)
        
    def _get_smol_A(self):
        # geometric-dependent part of the FM expression
        # by Smoluchowski, here for rectangular cross sections
        h, w = self.h, self.w
        return 2*(h**2*w*np.log(w/h+np.sqrt(1+(w/h)**2))
                  + h*w**2*np.log(h/w+np.sqrt(1+(h/w)**2))
                  - (h**2+w**2)**(3/2) / 3
                  + (h**3+w**3) / 3)
    
    def get_mdot_slip(self):
        """
        calulcates the convective mass flow including the slip influence
        for rectangular channels according to:
        
        # Jang, J., Wereley, S. (2004). Pressure distributions of 
        # gaseous slip flow in straight and uniform rectangular 
        # microchannels Microfluidics and Nanofluidics  1(1), 41-51.
        # https://dx.doi.org/10.1007/s10404-004-0005-8
        # eq. (10), replacing Kn_o by lambda_m_eff/h * p_m/p_o
            
        """
        
        mu = ph.get_mu(self.gas, self.T)
        M = ph.get_M(self.gas)
        if self.d_gas:
            mfp = rgd.mfp(self.T, self.d_gas, (self.pin+self.pout)/2)
        else:
            mfp = rgd.mfp_visc(self.T, (self.pin+self.pout)/2, self.gas)
        
        mfp_eff = 1/(1/mfp + 1/(2*self.h))
        
        asp_ratios = [2e-6, 2e-2, 0.2, 0.365, 0.4, 0.6, 0.8, 1]
        CP1_values = [-1.3333, -1.3165, -1.1653, -1.0267, -0.9975,
                      -0.8344, -0.6869, -0.5623]
        CP1 = np.interp(self.h/self.w, asp_ratios, CP1_values) 
        CPr_values = [6, 6.0150, 6.1692, 6.3520, 6.3984, 6.7482, 7.2816, 8]
        CPr = np.interp(self.h/self.w, asp_ratios, CPr_values)
        
        mdot_slip = ( self.w/2 * (self.h/2)**3 * self.pout**2 * (-CP1)
                     / (2*ph.R/M*self.T*mu*self.L) 
                     * ((self.pin/self.pout)**2 - 1 
                         + 2 * (2 - self.sigma_slip) / self.sigma_slip 
                        * mfp_eff/self.h * (self.pin+self.pout)/2 / self.pout 
                        * CPr * (self.pin/self.pout - 1)) )
        return mdot_slip
    
