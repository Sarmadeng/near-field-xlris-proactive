"""UPA figure: (a) exact vs separable range-focusing gain (Corollary);
(b) 2-D focal spot of the planar surface in the range/cross-range plane."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_upa import UPA, upa_fresnel

upa = UPA(Nx=64, Nz=64, fc=28e9)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.8))

# (a) exact vs separable
r0=8.0; rs=np.linspace(3,30,120)
sep=np.array([float(upa_fresnel(upa,r,r0)) for r in rs])
ax1.plot(rs,sep,"-",color="tab:purple",lw=1.7,label=r"separable $g_x g_z$ (Cor. 1)")
rex=rs[::9]; ex=[upa.focus_gain(r,r0) for r in rex]
ax1.plot(rex,ex,"o",color="tab:purple",ms=5,mfc="none",label="exact UPA sum")
ax1.axvline(r0,color="0.5",ls=":",lw=1,label=f"focus $r_0$={r0:.0f} m")
ax1.set_xlabel("user range $r$ [m]"); ax1.set_ylabel("normalised focusing gain")
ax1.set_title(f"(a) UPA {upa.Nx}×{upa.Nz} separability\n(max err $1.5\\times10^{{-3}}$)")
ax1.legend(fontsize=8); ax1.set_ylim(0,1.05)

# (b) 2-D focal spot: gain over (range r, cross-range y) focusing at (r0,0)
def gain2d(upa, r, y, r0):
    # user off-boresight by cross-range y at range r (in x-direction)
    X,Z=np.meshgrid(upa.x,upa.z,indexing="ij")
    du=np.sqrt((r)**2 + (X-y)**2 + Z**2)
    d0=np.sqrt((r0)**2 + X**2 + Z**2)
    return np.abs(np.mean(np.exp(1j*upa.k*(du-d0))))**2
rr=np.linspace(5,14,60); yy=np.linspace(-0.6,0.6,60)
G=np.array([[gain2d(upa,r,y,r0) for y in yy] for r in rr])
im=ax2.pcolormesh(yy*100, rr, G, shading="auto", cmap="viridis")
ax2.plot(0,r0,"r*",ms=12)
ax2.set_xlabel("cross-range [cm]"); ax2.set_ylabel("range $r$ [m]")
ax2.set_title("(b) 2-D focal spot (focus at $r_0$=8 m)")
fig.colorbar(im,ax=ax2,label="gain")
fig.tight_layout(); fig.savefig("../figs/fig_upa.pdf")
print(f"UPA {upa.Nx}x{upa.Nz}, Dx={upa.Dx:.3f} m, Rayleigh={upa.rayleigh:.1f} m")
print(f"max |exact-separable| over sweep = {np.max(np.abs(sep[::9][:len(ex)]-np.array(ex))):.4f}")
print("saved ../figs/fig_upa.pdf")
