"""Verify Theorem 1: exact near-field focusing gain vs Fresnel closed form, and
the depth-of-focus formula. Broadside focus (theta0=0)."""
import sys; sys.path.insert(0,".")
import numpy as np
from nf_core import XLRIS, fresnel_range_gain, depth_of_focus

ris = XLRIS(N=256, fc=28e9, d_frac=0.5)
print(f"N={ris.N} lambda={ris.lam*1e3:.2f} mm  aperture D={ris.D:.3f} m")
print(f"Rayleigh 2D^2/lam = {ris.rayleigh:.1f} m   Fresnel 0.5 sqrt(D^3/lam) = {ris.fresnel:.2f} m\n")

# 1) exact element-sum gain vs Fresnel closed form along range (broadside)
r0 = 20.0
rs = np.linspace(5.0, 120.0, 40)
print(f"focus r0={r0} m, theta0=0.  range   exact_g   fresnel_g   abs_err")
maxerr=0
for r in rs[::4]:
    ge = ris.focus_gain(0.0, r, 0.0, r0)
    gf = fresnel_range_gain(ris, r, r0)
    maxerr=max(maxerr, abs(ge-gf))
    print(f"  {r:7.1f}   {ge:7.4f}   {gf:8.4f}   {abs(ge-gf):.4f}")
# full-grid max error
ge_all = np.array([ris.focus_gain(0.0,r,0.0,r0) for r in rs])
gf_all = fresnel_range_gain(ris, rs, r0)
print(f"\nmax |exact-fresnel| over range sweep = {np.max(np.abs(ge_all-gf_all)):.4f}")

# 2) depth of focus: predicted vs measured 3-dB range interval
for r0 in [10.0, 20.0, 40.0, 80.0]:
    rn, rf, aL, beta = depth_of_focus(ris, r0, 0.5)
    # measured: scan range, find where exact gain crosses 0.5
    rr = np.linspace(2.0, min(ris.rayleigh, 10*r0), 4000)
    g = np.array([ris.focus_gain(0.0, r, 0.0, r0) for r in rr])
    above = rr[g>=0.5]
    m_near, m_far = (above.min(), above.max()) if above.size else (np.nan,np.nan)
    rf_s = "inf" if not np.isfinite(rf) else f"{rf:6.1f}"
    mf_s = f"{m_far:6.1f}" if np.isfinite(m_far) else "  n/a"
    print(f"r0={r0:5.1f}: predicted DoF=[{rn:5.1f},{rf_s}]  measured=[{m_near:5.1f},{mf_s}]  (1/beta={1/beta:5.1f} m)")
