"""
nf_core.py
================================================================================
Near-field XL-RIS array model and beam-focusing gain.

Geometry: uniform linear XL-RIS of N elements, spacing d, centred at origin.
Element m at x_m = (m - (N-1)/2) d,  m = 0..N-1.
A user at polar (range r, angle theta) [theta from broadside] has distance to
element m
        r_m(theta,r) = sqrt(r^2 + x_m^2 - 2 r x_m sin(theta)).
Near-field (spherical-wave) array response:
        a_m(theta,r) = exp(-j k r_m(theta,r)),   k = 2 pi / lambda.

The RIS focuses at (theta0, r0) with unit-modulus weights w_m = a_m(theta0,r0)^*.
Normalised beam-focusing gain (in [0,1], =1 at the focus):
        g(theta,r | theta0,r0) = | (1/N) sum_m a_m(theta,r) w_m |^2 .
The array gain is N * g (coherent gain N^2 at the focus / N normalisation choice).
"""
from __future__ import annotations
import numpy as np

C0 = 299_792_458.0

class XLRIS:
    def __init__(self, N=256, fc=28e9, d_frac=0.5):
        self.N = int(N)
        self.fc = float(fc)
        self.lam = C0 / self.fc
        self.d = d_frac * self.lam
        self.k = 2*np.pi / self.lam
        self.x = (np.arange(self.N) - (self.N-1)/2.0) * self.d   # element coords
        self.D = self.N * self.d                                 # aperture

    @property
    def rayleigh(self):
        """Rayleigh (Fraunhofer) distance 2 D^2 / lambda."""
        return 2.0 * self.D**2 / self.lam

    @property
    def fresnel(self):
        """Fresnel distance 0.5 sqrt(D^3/lambda) (lower near-field bound)."""
        return 0.5*np.sqrt(self.D**3/self.lam)

    def dist(self, theta, r):
        """r_m(theta,r); theta in rad, r in m. Returns (N,) array."""
        return np.sqrt(r**2 + self.x**2 - 2.0*r*self.x*np.sin(theta))

    def steer(self, theta, r):
        return np.exp(-1j*self.k*self.dist(theta, r))

    def focus_gain(self, theta, r, theta0, r0):
        """Normalised focusing gain g in [0,1]."""
        w = np.conj(self.steer(theta0, r0))
        a = self.steer(theta, r)
        return np.abs(np.vdot(w.conj(), a) / self.N)**2  # (1/N)|sum a_m w_m|^2 ...
    # note: vdot(w.conj(),a)=sum conj(conj w_m) a_m = sum w_m a_m


def fresnel_range_gain(ris: XLRIS, r, r0):
    """Second-order (Fresnel) closed form of the BROADSIDE range focusing gain.

    With theta=theta0=0, r_m ~ r + x_m^2/(2r). The residual phase after focusing
    is k x_m^2/2 (1/r - 1/r0). Converting the element sum to a Fresnel integral
    over the aperture gives
        g(r) = | (1/(2 a)) [ (C(a)+jS(a)) - (C(-a)+jS(-a)) ] |^2
             = ( C(a)^2 + S(a)^2 ) / a^2 ,
    with a = D/2 * sqrt( (1/lambda) |1/r - 1/r0| ),  C,S the Fresnel integrals.
    """
    from scipy.special import fresnel
    a = (ris.D/2.0) * np.sqrt((2.0/ris.lam) * np.abs(1.0/r - 1.0/r0))
    a = np.maximum(a, 1e-9)
    S, Cc = fresnel(a)           # scipy returns (S, C), pi/2 convention
    return (Cc**2 + S**2) / a**2


def depth_of_focus(ris: XLRIS, r0, level=0.5):
    """Closed-form 3-dB (or 'level') depth of focus in range about r0.

    g(r) depends on a(r) = (D/2) sqrt(|1/r-1/r0|/lambda); g=level at a=a_L where
    (C(a_L)^2+S(a_L)^2)/a_L^2 = level. Then |1/r-1/r0| = (2 a_L/D)^2 lambda, so
        r_near = 1/(1/r0 + beta),  r_far = 1/(1/r0 - beta),  beta=(2 a_L/D)^2 lam.
    r_far = infinity when 1/r0 <= beta  => finite DoF only for r0 < 1/beta.
    """
    from scipy.special import fresnel
    from scipy.optimize import brentq
    def g_of_a(a):
        S, Cc = fresnel(a); return (Cc**2+S**2)/a**2
    a_L = brentq(lambda a: g_of_a(a) - level, 1e-6, 5.0)
    beta = 2.0 * a_L**2 * ris.lam / ris.D**2
    r_near = 1.0/(1.0/r0 + beta)
    r_far = np.inf if (1.0/r0 - beta) <= 0 else 1.0/(1.0/r0 - beta)
    return r_near, r_far, a_L, beta
