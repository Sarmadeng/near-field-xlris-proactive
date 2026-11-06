"""
nf_upa.py — planar (UPA) near-field XL-RIS focusing and the separability corollary.

A UPA of Nx x Nz elements in the x-z plane, element (p,q) at (x_p, z_q, 0),
x_p=(p-(Nx-1)/2)d, z_q=(q-(Nz-1)/2)d. User on the boresight (z-normal) at range r.
Near-field focusing gain
    g = | (1/(Nx Nz)) sum_{p,q} exp(j k [ r_pq(r) - r_pq(r0) ]) |^2 .

Corollary (separability): at boresight the residual phase separates,
    psi_pq = (k/2)(x_p^2+z_q^2)(1/r-1/r0) = psi_p^x + psi_q^z,
so  g_UPA(r) = g_x(r) * g_z(r),  each a 1-D Fresnel focusing gain (Theorem 1) with
aperture D_x=Nx d and D_z=Nz d. For a square UPA, g_UPA=[g_1D]^2.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from nf_core import XLRIS, fresnel_range_gain, C0

class UPA:
    def __init__(self, Nx=64, Nz=64, fc=28e9, d_frac=0.5):
        self.Nx, self.Nz = Nx, Nz
        self.lam = C0/fc; self.d = d_frac*self.lam; self.k = 2*np.pi/self.lam
        self.x = (np.arange(Nx)-(Nx-1)/2)*self.d
        self.z = (np.arange(Nz)-(Nz-1)/2)*self.d
        self.Dx, self.Dz = Nx*self.d, Nz*self.d
        self.rayleigh = 2*max(self.Dx,self.Dz)**2/self.lam
    def dist(self, r):
        X,Z = np.meshgrid(self.x, self.z, indexing="ij")
        return np.sqrt(r**2 + X**2 + Z**2)          # boresight user
    def focus_gain(self, r, r0):
        psi = self.k*(self.dist(r)-self.dist(r0))
        return np.abs(np.mean(np.exp(1j*psi)))**2

def upa_fresnel(upa: UPA, r, r0):
    """Separable Fresnel prediction g_x(r)*g_z(r)."""
    rx = XLRIS(N=upa.Nx, fc=C0/upa.lam, d_frac=upa.d/upa.lam)
    rz = XLRIS(N=upa.Nz, fc=C0/upa.lam, d_frac=upa.d/upa.lam)
    return fresnel_range_gain(rx, r, r0)*fresnel_range_gain(rz, r, r0)

if __name__ == "__main__":
    upa = UPA(Nx=64, Nz=64, fc=28e9)
    print(f"UPA {upa.Nx}x{upa.Nz}={upa.Nx*upa.Nz} elems, Dx=Dz={upa.Dx:.3f} m, "
          f"Rayleigh={upa.rayleigh:.1f} m")
    r0 = 8.0
    print(f"\nfocus r0={r0} m (boresight):  r   exact_g   separable_gxgz   err")
    rs = np.linspace(3, 30, 28); exact=[]; sep=[]
    for r in rs:
        ge = upa.focus_gain(r, r0); gf = float(upa_fresnel(upa, r, r0))
        exact.append(ge); sep.append(gf)
    exact=np.array(exact); sep=np.array(sep)
    for i in range(0,len(rs),4):
        print(f"   {rs[i]:6.2f}   {exact[i]:7.4f}   {sep[i]:12.4f}   {abs(exact[i]-sep[i]):.4f}")
    print(f"\nmax |exact - separable| = {np.max(np.abs(exact-sep)):.4f}")
