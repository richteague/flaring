"""
Fit the radial profiles of (z, T) for a given molecule.
"""

import numpy as np
import scipy.constants as sc
from limepy.analysis.collisionalrates import ratefile
from scipy.special import erfc, erfinv


class molecule:

    def __init__(self, name):
        """Radiative transfer for the molecule."""
        self.rates = ratefile(name)
        return

    def Qrot(self, T):
        """Rotational partition function."""
        return T / self.rates.B0 + 1. / 3.

    def dV(self, T, vturb=0.0, mach=None):
        """Doppler linewidth in [m/s]."""
        vtherm = np.sqrt(2. * sc.k * T / self.rates.mu / sc.m_p)
        if vturb > 0.0 and mach is not None:
            raise ValueError("Only specify 'vturb' or 'mach'.")
        if vturb == 0.0 and mach is not None:
            vturb = mach * np.sqrt(sc.k * T / 2.34 / sc.m_p)
        return np.hypot(vtherm, vturb)

    def FWHM(self, T, vturb=0.0, mach=None):
        """Full width at half maximum of the line in [m/s]."""
        return np.sqrt(2. * np.log(2.)) * self.dV(T, vturb, mach)

    def N_tau(self, tau, T, Jl, vturb=0.0, mach=None):
        """Column density to reach an optical depth of tau in [/cm^2]."""
        transition = self.rates[Jl+1]
        nu, Au = transition.freq, transition.A
        gu = self.rates.levels[transition.i].g
        dV, Qr = self.FWHM(T, vturb, mach), self.Qrot(T)
        N = np.exp(sc.h * nu / sc.k / T) / (np.exp(sc.h * nu / sc.k / T) - 1.)
        N *= 8. * np.pi * tau * nu**3 * Qr * dV / sc.c**3 / Au / gu / 1e4
        return N

    def surface_density(self, z, T, Hgas, Jl, xmol=1e-4, Ndiss=1.3e21, tau=1.0,
                        vturb=0.0, mach=None):
        """Return surface density assuming a Gaussian profile [/cm^2]."""
        Nabove = Ndiss + self.N_tau(tau, T, Jl, vturb, mach) / xmol
        return 2 * Nabove / erfc(z / np.sqrt(2.) / Hgas)

    def emission_height(self, T, Hgas, log_sigma, Jl, xmol=1e-4, Ndiss=1.3e21,
                        tau=1.0, vturb=0.0, mach=None):
        """Return the emission height in [au] assuming Gaussian profile."""
        Nabove = Ndiss + self.N_tau(tau, T, Jl, vturb, mach) / xmol
        sigma = np.power(10, log_sigma)
        return np.sqrt(2.) * Hgas * erfinv((sigma - Nabove) / sigma)

    def density_profile(self, z, Hgas, log_sigma):
        """Density in [/cm^3] at a given height."""
        rho = np.power(10, log_sigma) / np.sqrt(2.) / Hgas
        return rho * np.exp(-0.5 * np.power(z / Hgas, 2.))
