"""Scenario sweeps:
(a) SE vs focused SNR (rho N) for the four schemes at v=15 (trained predictor);
(b) end-to-end SE vs reconfiguration latency, using the measured per-horizon
    predictor RMSE (horizon.json) for the proposed scheme and the true staleness
    for the reactive baseline."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS
from mobility import gauss_markov_trajectory
from predictor import AttnLSTM

ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
def gain(th,r,th_hat,r_hat,B=4):
    L=2**B; step=2*np.pi/L; ph=np.round(ris.k*ris.dist(th_hat,r_hat)/step)*step
    return np.abs(np.mean(np.exp(1j*(ris.k*ris.dist(th,r)-ph))))**2

def load():
    m=AttnLSTM(); m.load_state_dict(torch.load("predictor_model.pt")); m.eval()
    s=np.load("predictor_stats.npz"); return m,(s["xm"],s["xs"],s["ym"],s["ys"])

# ---- gather trained predictions at v=15 ----
def predictions(v=15.0,n_traj=80,seed=7):
    m,(xm,xs,ym,ys)=load(); L,H=64,40; sth=np.deg2rad(0.1)
    rng=np.random.default_rng(seed); out=[]
    for _ in range(n_traj):
        tr=gauss_markov_trajectory(200,speed=v,rng=rng); th,r=tr[:,0],tr[:,1]
        th_o=th+sth*rng.standard_normal(200); r_o=r*(1+0.01*rng.standard_normal(200))
        dth=np.diff(th_o,prepend=th_o[0]); dr=np.diff(r_o,prepend=r_o[0])
        feat=np.stack([th_o,r_o,dth,dr],1); idx=np.arange(L,200-H)
        W=np.stack([feat[n-L:n] for n in idx]); Wn=((W-xm)/xs).astype(np.float32)
        with torch.no_grad(): pr=m(torch.tensor(Wn)).numpy()*ys+ym
        for j,n in enumerate(idx): out.append((th[n+H],r[n+H],pr[j,0],pr[j,1],th_o[n],r_o[n]))
    return np.array(out)

S=predictions()
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.8,3.8))

# (a) SE vs SNR
snrs=np.arange(0,31,3)
g_prop=np.array([gain(a,b,c,d) for a,b,c,d,e,f in S])
g_reac=np.array([gain(a,b,e,f) for a,b,c,d,e,f in S])
g_ff  =np.array([gain(a,b,e,1e9) for a,b,c,d,e,f in S])
for g,lab,col,mk in [(np.ones_like(g_prop),"Genie","0.3","" ),
                     (g_prop,"Proposed","tab:red","o"),
                     (g_reac,"Reactive","tab:blue","s"),
                     (g_ff,"Far-field","tab:green","^")]:
    se=[np.mean(np.log2(1+10**(s/10.0)*g)) for s in snrs]
    ax1.plot(snrs,se,marker=mk,ls=("--" if lab=="Genie" else "-"),color=col,lw=1.6,ms=5,label=lab)
ax1.set_xlabel("focused SNR $\\rho N$ [dB]"); ax1.set_ylabel("mean effective SE [bit/s/Hz]")
ax1.set_title("(a) SE vs SNR (v=15 m/s, N=256)"); ax1.legend(fontsize=8)

# (b) SE vs latency: proposed via measured RMSE(H); reactive via true staleness
hz=json.load(open("horizon.json"))   # [H, ms, rmse_theta_deg, rmse_r_m]
rhoN=10**(20/10.0); rng=np.random.default_rng(3)
lat=[]; se_p=[]; se_r=[]; se_g=[]
# pre-generate trajectories for reactive staleness
trs=[gauss_markov_trajectory(260,speed=15.0,rng=rng) for _ in range(60)]
for H,ms,rt,rr in hz:
    H=int(H)
    # proposed: Gaussian error with measured RMSE
    N=6000
    th=np.deg2rad(rng.uniform(-45,45,N)); r=rng.uniform(10,40,N)
    thp=th+np.deg2rad(rt)*rng.standard_normal(N); rp=r+rr*rng.standard_normal(N)
    gp=np.array([gain(th[i],r[i],thp[i],rp[i]) for i in range(0,N,3)])
    se_p.append(np.mean(np.log2(1+rhoN*gp)))
    # reactive: focus at position H slots ago (staleness)
    gr=[]
    for tr in trs:
        thf,rf=tr[:,0],tr[:,1]
        for n in range(80,260-H,7):
            gr.append(gain(thf[n],rf[n],thf[n-H],rf[n-H]))
    se_r.append(np.mean([np.log2(1+rhoN*g) for g in gr]))
    se_g.append(np.log2(1+rhoN*gain(0.0,20.0,0.0,20.0)))
    lat.append(ms)
ax2.plot(lat,se_g,"k--",lw=1.4,label="genie")
ax2.plot(lat,se_p,"o-",color="tab:red",lw=1.7,ms=5,label="proposed")
ax2.plot(lat,se_r,"s-",color="tab:blue",lw=1.7,ms=5,label="reactive")
ax2.set_xlabel("reconfiguration latency $\\tau_{lat}$ [ms]"); ax2.set_ylabel("mean effective SE [bit/s/Hz]")
ax2.set_title("(b) SE vs latency (v=15 m/s)"); ax2.legend(fontsize=8); ax2.set_ylim(0,7)
fig.tight_layout(); fig.savefig("../figs/fig_scenarios.pdf")
print("SE vs SNR done; SE vs latency:")
for i,(H,ms,rt,rr) in enumerate(hz):
    print(f"  tau={ms:4.0f} ms: proposed={se_p[i]:.2f} reactive={se_r[i]:.2f} genie={se_g[i]:.2f}")
print("saved ../figs/fig_scenarios.pdf")
