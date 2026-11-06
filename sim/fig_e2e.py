"""Main results figure: (a) effective SE vs user speed for the four schemes;
(b) SE vs user range showing the near-field focusing advantage vanishing beyond
the effective focusing range. Good predictor (sig_theta=0.1 deg, sig_r=0.5%)."""
import sys; sys.path.insert(0,".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS, fresnel_range_gain
from e2e_mobility import run, gain

ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.8,3.8))
order=["genie","proposed","reactive","farfield"]
lab={"genie":"Genie (perfect focus)","proposed":"Proposed (predict-then-compute)",
     "reactive":"Reactive near-field","farfield":"Far-field codebook"}
col={"genie":"0.3","proposed":"tab:red","reactive":"tab:blue","farfield":"tab:green"}
mk ={"genie":"","proposed":"o","reactive":"s","farfield":"^"}
ls ={"genie":"--","proposed":"-","reactive":"-","farfield":"-"}

# (a) SE vs speed
speeds=[1,5,10,15,20,25,30]
res={k:[] for k in order}; err={k:[] for k in order}
for v in speeds:
    r=run(v,ris,sigth_deg=0.1,sigr_frac=0.005,seeds=12)
    for k in order: res[k].append(r[k][0]); err[k].append(r[k][1])
for k in order:
    ax1.errorbar(speeds,res[k],yerr=err[k],marker=mk[k],ls=ls[k],color=col[k],
                 lw=1.7,ms=5,capsize=2,label=lab[k])
ax1.set_xlabel("user speed $v$ [m/s]"); ax1.set_ylabel("effective SE [bit/s/Hz]")
ax1.set_title("(a) Robustness to mobility\n($\\tau_{lat}$=20 ms, $r_0$=20 m, $B$=4)")
ax1.legend(fontsize=7.5,loc="center right"); ax1.set_ylim(0,7.2)

# (b) SE vs range at fixed speed 15 m/s: proposed vs far-field codebook
ranges=np.linspace(6,120,22)
rhoN=10**(20/10); B=4
se_prop=[]; se_ff=[]
for r0 in ranges:
    rr=run(15.0,ris,r0=r0,sigth_deg=0.1,sigr_frac=0.005,seeds=8)
    se_prop.append(rr["proposed"][0]); se_ff.append(rr["farfield"][0])
ax2.plot(ranges,se_prop,"o-",color="tab:red",lw=1.7,ms=4,label="Proposed (near-field focus)")
ax2.plot(ranges,se_ff,"^-",color="tab:green",lw=1.7,ms=4,label="Far-field codebook")
ax2.axvline(50.5,color="0.5",ls=":",lw=1.2,label="eff. focus range $1/\\beta$")
ax2.set_xlabel("user range $r_0$ [m]"); ax2.set_ylabel("effective SE [bit/s/Hz]")
ax2.set_title("(b) Near-field focusing gain vs range\n(v=15 m/s)")
ax2.legend(fontsize=8); ax2.set_ylim(0,7.2)
fig.tight_layout(); fig.savefig("../figs/fig_e2e_results.pdf")

# print summary numbers for the manuscript
print("SE vs speed (good predictor):")
for i,v in enumerate(speeds):
    print(f"  v={v:2d}: genie={res['genie'][i]:.2f} proposed={res['proposed'][i]:.2f} "
          f"reactive={res['reactive'][i]:.2f} farfield={res['farfield'][i]:.2f}")
g30=res['proposed'][-1]/res['reactive'][-1]
print(f"\nproposed/reactive at 30 m/s = {g30:.2f}x ; proposed/genie at 30 = {res['proposed'][-1]/res['genie'][-1]*100:.1f}%")
print(f"proposed/farfield at r0=20 (v15): see panel b; near-field gain near r0=20")
print("saved ../figs/fig_e2e_results.pdf")
