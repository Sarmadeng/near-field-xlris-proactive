"""Design sweeps with the trained predictor (fixed v=15 m/s):
(a) SE vs phase-quantisation bits B; (b) SE vs surface size N.
Both reuse the trained predictor's (theta,r) forecasts; only the focusing array
(B or N) changes, isolating Theorem 2's quantisation and aperture dependence."""
import sys; sys.path.insert(0,".")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nf_core import XLRIS
from mobility import gauss_markov_trajectory
from predictor import AttnLSTM

def load():
    m=AttnLSTM(); m.load_state_dict(torch.load("predictor_model.pt")); m.eval()
    s=np.load("predictor_stats.npz"); return m,(s["xm"],s["xs"],s["ym"],s["ys"])

def gain(ris,th,r,th_hat,r_hat,B):
    L=2**B; step=2*np.pi/L; ph=np.round(ris.k*ris.dist(th_hat,r_hat)/step)*step
    return np.abs(np.mean(np.exp(1j*(ris.k*ris.dist(th,r)-ph))))**2

def predictions(v=15.0,n_traj=60,seed=7):
    m,(xm,xs,ym,ys)=load(); L,H=64,40; sth=np.deg2rad(0.1)
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(n_traj):
        tr=gauss_markov_trajectory(200,speed=v,rng=rng); th,r=tr[:,0],tr[:,1]
        th_o=th+sth*rng.standard_normal(200); r_o=r*(1+0.01*rng.standard_normal(200))
        dth=np.diff(th_o,prepend=th_o[0]); dr=np.diff(r_o,prepend=r_o[0])
        feat=np.stack([th_o,r_o,dth,dr],1); idx=np.arange(L,200-H)
        W=np.stack([feat[n-L:n] for n in idx]); Wn=((W-xm)/xs).astype(np.float32)
        with torch.no_grad(): pr=m(torch.tensor(Wn)).numpy()*ys+ym
        for j,n in enumerate(idx): samples.append((th[n+H],r[n+H],pr[j,0],pr[j,1]))
    return np.array(samples)

S=predictions(); rhoN=10**(20/10.0)
# (a) SE vs B at N=256
risB=XLRIS(N=256)
Bs=[1,2,3,4,5,6]; seB=[]
for B in Bs:
    g=np.array([gain(risB,a,b,c,d,B) for a,b,c,d in S]); seB.append(np.mean(np.log2(1+rhoN*g)))
# (b) SE vs N at B=4 (predictor errors fixed)
Ns=[64,128,256,384,512]; seN=[]; genN=[]
for N in Ns:
    rr=XLRIS(N=N)
    g=np.array([gain(rr,a,b,c,d,4) for a,b,c,d in S]); seN.append(np.mean(np.log2(1+rhoN*g)))
    gg=np.array([gain(rr,a,b,a,b,4) for a,b,c,d in S]); genN.append(np.mean(np.log2(1+rhoN*gg)))

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,3.6))
ax1.plot(Bs,seB,"o-",color="tab:red",lw=1.8,ms=6)
ax1.set_xlabel("phase quantisation bits $B$"); ax1.set_ylabel("mean effective SE [bit/s/Hz]")
ax1.set_title("(a) Effect of phase quantisation\n(N=256, v=15 m/s)"); ax1.set_ylim(0,7)
ax2.plot(Ns,genN,"s--",color="0.5",lw=1.6,ms=5,label="genie (perfect focus)")
ax2.plot(Ns,seN,"o-",color="tab:red",lw=1.8,ms=6,label="proposed (trained)")
ax2.set_xlabel("surface size $N$"); ax2.set_ylabel("mean effective SE [bit/s/Hz]")
ax2.set_title("(b) Effect of surface size\n(B=4, v=15 m/s)"); ax2.legend(fontsize=8); ax2.set_ylim(0,9)
fig.tight_layout(); fig.savefig("../figs/fig_sweeps.pdf")
print("SE vs B:", {b:round(s,2) for b,s in zip(Bs,seB)})
print("SE vs N (proposed):", {n:round(s,2) for n,s in zip(Ns,seN)})
print("SE vs N (genie):   ", {n:round(s,2) for n,s in zip(Ns,genN)})
print("proposed/genie ratio vs N:", {n:round(p/g,2) for n,p,g in zip(Ns,seN,genN)})
print("saved ../figs/fig_sweeps.pdf")
