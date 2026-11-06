"""Theorem 1 figure: (a) exact vs Fresnel range-focusing gain for several focus
distances; (b) 3-dB depth of focus and effective focusing range vs r0."""
import sys; sys.path.insert(0,".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS, fresnel_range_gain, depth_of_focus

ris = XLRIS(N=256, fc=28e9, d_frac=0.5)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

# (a) gain vs range
rs = np.linspace(4.0, 140.0, 500)
cols = ["tab:blue","tab:orange","tab:green","tab:red"]
for r0,c in zip([10.0,20.0,40.0,80.0], cols):
    gf = fresnel_range_gain(ris, rs, r0)
    ax1.plot(rs, gf, "-", color=c, lw=1.6, label=f"$r_0$={r0:.0f} m")
    rex = rs[::28]
    ge = [ris.focus_gain(0.0, r, 0.0, r0) for r in rex]
    ax1.plot(rex, ge, "o", color=c, ms=4, mfc="none")
ax1.axhline(0.5, color="0.5", ls=":", lw=1)
ax1.set_xlabel("user range $r$ [m]"); ax1.set_ylabel("normalised focusing gain $g(r)$")
ax1.set_title("(a) Range focusing: lines = Fresnel (Thm 1),\n markers = exact element sum")
ax1.legend(fontsize=8, title="focus distance")
ax1.set_ylim(0,1.05)

# (b) depth of focus vs r0
r0s = np.linspace(6.0, 120.0, 120)
rn, rf = [], []
for r0 in r0s:
    a,b,aL,beta = depth_of_focus(ris, r0, 0.5)
    rn.append(a); rf.append(min(b, ris.rayleigh))
rn=np.array(rn); rf=np.array(rf)
ax2.fill_between(r0s, rn, rf, alpha=0.25, color="tab:blue", label="3-dB focus region")
ax2.plot(r0s, r0s, "k--", lw=1, label="$r=r_0$")
_,_,aL,beta = depth_of_focus(ris, 20.0, 0.5)
ax2.axvline(1.0/beta, color="tab:red", ls=":", lw=1.4, label=f"eff. focus range $1/\\beta$={1/beta:.0f} m")
ax2.set_xlabel("focus distance $r_0$ [m]"); ax2.set_ylabel("range [m]")
ax2.set_title("(b) 3-dB depth of focus\n(unbounded beyond $1/\\beta$)")
ax2.legend(fontsize=8); ax2.set_ylim(0, ris.rayleigh)
fig.tight_layout(); fig.savefig("../figs/fig_thm1_focusing.pdf")
print("saved ../figs/fig_thm1_focusing.pdf")
print(f"effective focusing range 1/beta = {1/beta:.1f} m  (Rayleigh {ris.rayleigh:.0f} m, ratio {1/beta/ris.rayleigh:.3f})")
