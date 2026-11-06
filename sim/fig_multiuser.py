"""Multi-user near-field scenario: RANGE-DOMAIN user separation.
Two users share the same angle (theta=0) but lie at different ranges. A near-field
surface focused on user 1 leaks little to user 2 (range selectivity, Theorem 1),
so the two are separable in the range domain; a far-field codebook (angle only)
cannot separate them. We plot the user isolation and the two-user TDMA sum rate
versus range separation."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS, fresnel_range_gain

ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
def gain(th,r,th_hat,r_hat,B=4):
    L=2**B; step=2*np.pi/L; ph=np.round(ris.k*ris.dist(th_hat,r_hat)/step)*step
    return np.abs(np.mean(np.exp(1j*(ris.k*ris.dist(th,r)-ph))))**2

r1=10.0; r2s=np.linspace(10.5,40,60); rhoN=10**(20/10.0)
iso_nf=[]; iso_ff=[]; sr_nf=[]; sr_ff=[]
for r2 in r2s:
    # near-field: focus at user1 (0,r1); leakage to user2 = gain(0,r2 | 0,r1)
    g11=gain(0.0,r1,0.0,r1); g21=gain(0.0,r2,0.0,r1)
    iso_nf.append(10*np.log10(g11/max(g21,1e-6)))
    # far-field: focus at angle 0, range=inf; both users same angle -> near-equal gain
    g11f=gain(0.0,r1,0.0,1e9); g21f=gain(0.0,r2,0.0,1e9)
    iso_ff.append(10*np.log10(g11f/max(g21f,1e-6)))
    # two-user TDMA sum rate with inter-user leakage as interference (half slot each)
    # near-field: user1 focus (signal g11, interference to self none); user2 focus (g22,leak g12)
    g22=gain(0.0,r2,0.0,r2); g12=gain(0.0,r1,0.0,r2)
    sinr1=rhoN*g11/(1+rhoN*g21); sinr2=rhoN*g22/(1+rhoN*g12)
    sr_nf.append(0.5*np.log2(1+sinr1)+0.5*np.log2(1+sinr2))
    # far-field: cannot range-separate; focus angle only, both share pattern
    sinr1f=rhoN*g11f/(1+rhoN*g21f); sinr2f=rhoN*g21f/(1+rhoN*g11f)
    sr_ff.append(0.5*np.log2(1+sinr1f)+0.5*np.log2(1+sinr2f))

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.8))
ax1.plot(r2s-r1,iso_nf,"-",color="tab:red",lw=1.8,label="near-field focus")
ax1.plot(r2s-r1,iso_ff,"--",color="tab:green",lw=1.8,label="far-field codebook")
ax1.axhline(10,color="0.5",ls=":",lw=1,label="10 dB isolation")
ax1.set_xlabel("range separation $r_2-r_1$ [m]"); ax1.set_ylabel("user isolation [dB]")
ax1.set_title("(a) Range-domain user isolation\n(same angle, $r_1$=10 m)"); ax1.legend(fontsize=8)
ax2.plot(r2s-r1,sr_nf,"-",color="tab:red",lw=1.8,label="near-field (proposed)")
ax2.plot(r2s-r1,sr_ff,"--",color="tab:green",lw=1.8,label="far-field codebook")
ax2.set_xlabel("range separation $r_2-r_1$ [m]"); ax2.set_ylabel("two-user TDMA sum rate [bit/s/Hz]")
ax2.set_title("(b) Two-user sum rate\n(same angle)"); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figs/fig_multiuser.pdf")
i=np.argmin(np.abs((r2s-r1)-10))
print(f"at range sep 10 m: isolation near-field={iso_nf[i]:.1f} dB, far-field={iso_ff[i]:.1f} dB")
print(f"sum rate near-field={sr_nf[i]:.2f}, far-field={sr_ff[i]:.2f} bit/s/Hz")
print("saved ../figs/fig_multiuser.pdf")
