"""
End-to-end spectral efficiency using the TRAINED Attention-LSTM predictor.
For each speed we generate fresh test trajectories, run the trained model to
predict (theta,r) H slots ahead at every step, compute the EXACT near-field
focusing gain (nf_core) for the proposed/reactive/far-field schemes, and the SE.
Genie = perfect focus at the true future position.
"""
import sys; sys.path.insert(0,".")
import numpy as np, torch
from nf_core import XLRIS, fresnel_range_gain
from mobility import gauss_markov_trajectory
from predictor import AttnLSTM

def load_model(path="predictor_model.pt", stats="predictor_stats.npz"):
    m = AttnLSTM(); m.load_state_dict(torch.load(path)); m.eval()
    s = np.load(stats)
    return m, (s["xm"], s["xs"], s["ym"], s["ys"])

def gain(ris, th, r, th_hat, r_hat, B):
    L=2**B; step=2*np.pi/L
    ph=np.round(ris.k*ris.dist(th_hat,r_hat)/step)*step
    return np.abs(np.mean(np.exp(1j*(ris.k*ris.dist(th,r)-ph))))**2

def run_speed(v, ris, model, stt, B=4, rho_N_dB=20.0, L=64, H=40,
              obs_sig_th_deg=0.1, obs_sig_r_frac=0.01, n_traj=40, n_steps=200, seed=7):
    xm,xs,ym,ys = stt
    rhoN=10**(rho_N_dB/10.0); sth=np.deg2rad(obs_sig_th_deg)
    rng=np.random.default_rng(seed+int(v))
    se={k:[] for k in ["genie","proposed","reactive","farfield"]}
    for _ in range(n_traj):
        tr=gauss_markov_trajectory(n_steps, speed=float(v), rng=rng)
        th,r=tr[:,0],tr[:,1]
        th_o=th+sth*rng.standard_normal(n_steps)
        r_o =r*(1.0+obs_sig_r_frac*rng.standard_normal(n_steps))
        dth=np.diff(th_o,prepend=th_o[0]); dr=np.diff(r_o,prepend=r_o[0])
        feat=np.stack([th_o,r_o,dth,dr],axis=1)
        # batch all windows of this trajectory
        idx=np.arange(L, n_steps-H)
        W=np.stack([feat[n-L:n] for n in idx])          # (M,L,4)
        Wn=((W-xm)/xs).astype(np.float32)
        with torch.no_grad():
            pr=model(torch.tensor(Wn)).numpy()*ys+ym     # (M,2) predicted (th,r) at +H
        for j,n in enumerate(idx):
            tht,rt=th[n+H],r[n+H]                         # TRUE future
            # genie
            se["genie"].append(np.log2(1+rhoN*gain(ris,tht,rt,tht,rt,B)))
            # proposed: LSTM prediction
            se["proposed"].append(np.log2(1+rhoN*gain(ris,tht,rt,pr[j,0],pr[j,1],B)))
            # reactive: last noisy measurement (no look-ahead)
            se["reactive"].append(np.log2(1+rhoN*gain(ris,tht,rt,th_o[n],r_o[n],B)))
            # far-field: last angle, range at infinity
            se["farfield"].append(np.log2(1+rhoN*gain(ris,tht,rt,th_o[n],1e9,B)))
    return {k:(float(np.mean(x)),float(np.std(x))) for k,x in se.items()}

if __name__=="__main__":
    ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
    model,stt=load_model()
    speeds=[1,5,10,15,20,25,30]
    order=["genie","proposed","reactive","farfield"]
    print(f"{'v':>4s} "+" ".join(f"{k:>16s}" for k in order))
    data={k:[] for k in order}
    for v in speeds:
        res=run_speed(v,ris,model,stt)
        print(f"{v:4d} "+" ".join(f"{res[k][0]:7.3f}+-{res[k][1]:4.2f}" for k in order))
        for k in order: data[k].append(res[k])
    np.save("e2e_trained_data.npy",{"speeds":speeds,"data":data},allow_pickle=True)
    i30=speeds.index(30)
    print(f"\nproposed/genie = {data['proposed'][i30][0]/data['genie'][i30][0]*100:.1f}% at 30 m/s")
    print(f"proposed/reactive at 30 = {data['proposed'][i30][0]/data['reactive'][i30][0]:.2f}x")
    print("saved e2e_trained_data.npy")
