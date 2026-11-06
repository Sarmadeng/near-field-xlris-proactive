"""Final results figure with the TRAINED predictor:
(a) mean effective SE vs speed (proposed/reactive/far-field/genie), with the
    reactive->proposed crossover; (b) SE vs range (near-field advantage)."""
import sys; sys.path.insert(0,".")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS
from e2e_trained import load_model, run_speed, gain
from mobility import gauss_markov_trajectory

ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
model,stt=load_model()
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.8,3.8))
order=["genie","proposed","reactive","farfield"]
lab={"genie":"Genie (perfect focus)","proposed":"Proposed (Attention-LSTM)",
     "reactive":"Reactive near-field","farfield":"Far-field codebook"}
col={"genie":"0.3","proposed":"tab:red","reactive":"tab:blue","farfield":"tab:green"}
mk ={"genie":"","proposed":"o","reactive":"s","farfield":"^"}
ls ={"genie":"--","proposed":"-","reactive":"-","farfield":"-"}

# (a) SE vs speed (more trajectories -> stable mean; plot mean +- SEM)
speeds=[1,5,10,15,20,25,30]
res={k:[] for k in order}; sem={k:[] for k in order}
for v in speeds:
    r=run_speed(v,ris,model,stt,n_traj=120)
    for k in order:
        res[k].append(r[k][0]); sem[k].append(r[k][1]/np.sqrt(120*96))
for k in order:
    ax1.errorbar(speeds,res[k],yerr=sem[k],marker=mk[k],ls=ls[k],color=col[k],
                 lw=1.7,ms=5,capsize=2,label=lab[k])
ax1.set_xlabel("user speed $v$ [m/s]"); ax1.set_ylabel("mean effective SE [bit/s/Hz]")
ax1.set_title("(a) Robustness to mobility (trained predictor)\n($\\tau_{lat}$=20 ms, $r_0\\!\\approx$20 m, $B$=4)")
ax1.legend(fontsize=7.5,loc="upper right"); ax1.set_ylim(0,7.2)

# (b) Near-field focusing advantage vs range (Theorem 1), isolated from the
# predictor: near-field focus at the true (theta,r) vs far-field codebook (true
# angle, range=infinity). Deterministic consequence of the focusing gain.
rhoN=10**(20/10.0)
bands=[12,16,20,26,32,40,55,75,100,140,200]
sp=[];sf=[]
for b in bands:
    gp=gain(ris,0.0,float(b),0.0,float(b),4)          # near-field, perfect focus
    gf=gain(ris,0.0,float(b),0.0,1e9,4)               # far-field codebook (range=inf)
    sp.append(np.log2(1+rhoN*gp)); sf.append(np.log2(1+rhoN*gf))
ax2.plot(bands,sp,"o-",color="tab:red",lw=1.7,ms=4,label="Proposed (near-field focus)")
ax2.plot(bands,sf,"^-",color="tab:green",lw=1.7,ms=4,label="Far-field codebook")
ax2.axvline(50.5,color="0.5",ls=":",lw=1.2,label="eff. focus range $1/\\beta$")
ax2.set_xlabel("nominal user range $r_0$ [m]"); ax2.set_ylabel("mean effective SE [bit/s/Hz]")
ax2.set_title("(b) Near-field focusing advantage vs range\n(v=15 m/s)")
ax2.legend(fontsize=8); ax2.set_ylim(0,7.2)
fig.tight_layout(); fig.savefig("../figs/fig_e2e_trained.pdf")

print("SE vs speed (trained predictor):")
for i,v in enumerate(speeds):
    print(f"  v={v:2d}: genie={res['genie'][i]:.2f} proposed={res['proposed'][i]:.2f} "
          f"reactive={res['reactive'][i]:.2f} farfield={res['farfield'][i]:.2f}")
i30=speeds.index(30)
print(f"proposed/genie @30 = {res['proposed'][i30]/res['genie'][i30]*100:.1f}%  "
      f"proposed/reactive @30 = {res['proposed'][i30]/res['reactive'][i30]:.2f}x")
# crossover speed
cr=[v for i,v in enumerate(speeds) if res['proposed'][i]>res['reactive'][i]]
print(f"proposed exceeds reactive from v>= {cr[0] if cr else 'n/a'} m/s")
np.save("e2e_trained_fig.npy",{"speeds":speeds,"res":res,"bands":bands,"sp":sp,"sf":sf},allow_pickle=True)
print("saved ../figs/fig_e2e_trained.pdf")
