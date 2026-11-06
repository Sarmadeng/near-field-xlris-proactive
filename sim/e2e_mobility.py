"""
End-to-end mobility simulation: effective spectral efficiency of near-field
XL-RIS beam focusing under user motion and a realistic reconfiguration latency.

The binding impairment in mobile RIS control is the reconfiguration LATENCY
tau_lat (one scheduling slot, 10-50 ms [Abdulhussien & Ibrahim, PhysComm 2026]):
a configuration computed from a measured position is only applied tau_lat later,
by which time the user has moved -> staleness loss that grows with speed.

Schemes
  genie     : perfect instantaneous focus (upper bound).
  proposed  : predict-then-compute. Attention-LSTM predicts (theta,r) tau_lat
              ahead (error modelled from its reported sub-degree accuracy);
              near-field focusing phases computed in closed form (Theorem 1).
  reactive  : near-field focus at the last MEASURED position, applied tau_lat
              late -> staleness = motion over tau_lat.
  farfield  : angle-only codebook (focus at infinity); near-field defocus loss
              (Theorem 1), and still applied tau_lat late in angle.

Effective SE = log2(1 + rho*N * g) averaged over the trajectory and seeds.
B-bit phase quantisation throughout. Report mean +- std over seeds.
"""
import sys; sys.path.insert(0,".")
import numpy as np
from nf_core import XLRIS, fresnel_range_gain

def gain(ris, th, r, th_hat, r_hat, B):
    L=2**B; step=2*np.pi/L
    ph=np.round(ris.k*ris.dist(th_hat,r_hat)/step)*step
    return np.abs(np.mean(np.exp(1j*(ris.k*ris.dist(th,r)-ph))))**2

def run(v, ris, rho_N_dB=20.0, B=4, tau_lat=20e-3, T_slot=0.5e-3,
        r0=20.0, theta0_deg=15.0, sigth_deg=0.2, sigr_frac=0.01,
        n_slots=300, seeds=10):
    rhoN=10**(rho_N_dB/10.0); th0=np.deg2rad(theta0_deg)
    out={k:[] for k in ["genie","proposed","reactive","farfield"]}
    for s in range(seeds):
        rng=np.random.default_rng(2000+s)
        head=rng.uniform(0,2*np.pi)              # random heading per trajectory
        vt=v*np.cos(head); vr=v*np.sin(head)
        r,th=r0,th0; se={k:[] for k in out}
        for n in range(n_slots):
            om=vt/max(r,1e-3)
            # position at apply-time = now + tau_lat (where the beam must point)
            th_future=th+om*tau_lat; r_future=r+vr*tau_lat
            # genie: focus exactly at future pos (only quant)
            se["genie"].append(np.log2(1+rhoN*gain(ris,th_future,r_future,th_future,r_future,B)))
            # proposed: predict future pos with LSTM error
            thp=th_future+rng.normal(0,np.deg2rad(sigth_deg))
            rp =r_future+rng.normal(0,sigr_frac*r_future)
            se["proposed"].append(np.log2(1+rhoN*gain(ris,th_future,r_future,thp,rp,B)))
            # reactive: focus at measured NOW position, applied tau_lat late
            se["reactive"].append(np.log2(1+rhoN*gain(ris,th_future,r_future,th,r,B)))
            # far field: angle at NOW (codebook), range=inf -> near-field defocus
            g_ang=gain(ris,th_future,r_future,th,1e9,B)   # wrong range + stale angle
            # combine with pure near-field range-defocus reference (focus r=inf):
            se["farfield"].append(np.log2(1+rhoN*g_ang))
            th+=om*T_slot; r+=vr*T_slot
        for k in out: out[k].append(np.mean(se[k]))
    return {k:(np.mean(x),np.std(x)) for k,x in out.items()}

if __name__=="__main__":
    ris=XLRIS(N=256,fc=28e9,d_frac=0.5)
    speeds=[1,5,10,15,20,25,30]
    order=["genie","proposed","reactive","farfield"]
    print("reconfig latency tau=20 ms, B=4, rhoN=20 dB, r0=20 m, sig_theta=0.2deg, sig_r=1%")
    print(f"{'v[m/s]':>7s} "+" ".join(f"{k:>16s}" for k in order))
    data={k:[] for k in order}
    for v in speeds:
        res=run(v,ris)
        print(f"{v:7.0f} "+" ".join(f"{res[k][0]:7.3f}+-{res[k][1]:4.2f}" for k in order))
        for k in order: data[k].append(res[k])
    np.save("e2e_data.npy",{"speeds":speeds,"data":data},allow_pickle=True)
    print("saved e2e_data.npy")
