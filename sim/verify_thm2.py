"""Verify Theorem 2: rate-loss bound of predict-then-compute focusing under
(angle,range) prediction error and B-bit phase quantisation.

Exact gain with focus at predicted (th_hat, r_hat), quantised phases:
    phi_m = Q_B( k r_m(th_hat, r_hat) ),   Q_B rounds to 2^B levels on [0,2pi)
    g = | (1/N) sum_m exp( j[ k r_m(th,r) - phi_m ] ) |^2
Rate loss vs genie (perfect focus, g=1):  dR = log2((1+gamma)/(1+gamma g)) <= -log2(g).

Bound:  g >= (1 - V/2)^2,  V = (1/N) sum_m psi_m^2  (mean-square phase error),
and V <= 3(V_a + V_r + V_q) with
    V_q = pi^2/(3 4^B)                          (quantisation)
    V_a = k^2 cos^2(th) dtheta^2 M2,  M2=(1/N)sum x_m^2
    V_r = (k^2/4) (1/r-1/rhat)^2 M4,  M4=(1/N)sum x_m^4
"""
import sys; sys.path.insert(0,".")
import numpy as np
from nf_core import XLRIS

def quantize_phase(ph, B):
    L = 2**B
    step = 2*np.pi/L
    return np.round(ph/step)*step

def exact_gain(ris, th, r, th_hat, r_hat, B):
    ph = quantize_phase(ris.k*ris.dist(th_hat, r_hat), B) if B is not None else ris.k*ris.dist(th_hat,r_hat)
    psi = ris.k*ris.dist(th, r) - ph
    return np.abs(np.mean(np.exp(1j*psi)))**2

def V_components(ris, th, r, th_hat, r_hat, B):
    # gain depends on the VARIANCE of the element phase (common piston cancels)
    M2 = np.mean(ris.x**2); M4 = np.mean(ris.x**4)
    dth = th - th_hat
    V_a = (ris.k*np.cos(th))**2 * dth**2 * M2            # E[x^2]; angle phase has zero mean
    V_r = (ris.k**2/4.0) * (1.0/r - 1.0/r_hat)**2 * (M4 - M2**2)  # central 4th moment
    V_q = (np.pi**2/(3.0*4**B)) if B is not None else 0.0
    return V_a, V_r, V_q

if __name__ == "__main__":
    ris = XLRIS(N=256, fc=28e9, d_frac=0.5)
    rng = np.random.default_rng(0)
    gamma = 10**(10/10)   # 10 dB focused SNR per-element-normalised (rho N absorbed)

    # ---- (i) single-source checks: does V_x predict avg psi^2 ? ----
    print("Single-source mean-square phase error: analytic V vs measured avg(psi^2)")
    th0, r0 = np.deg2rad(10.0), 20.0
    # angle only
    dth = np.deg2rad(0.3)
    psi = ris.k*(ris.dist(th0, r0) - ris.dist(th0+dth, r0))  # focus at th0, user at th0?
    # measure with focus at (th0,r0), user at (th0+dth, r0):
    psi = ris.k*(ris.dist(th0+dth, r0) - ris.dist(th0, r0))
    Va,_,_ = V_components(ris, th0+dth, r0, th0, r0, None)
    print(f"  angle  dtheta=0.3deg:  V_a={Va:.4f}  meas_var={np.var(psi):.4f}")
    dr = 2.0
    psi = ris.k*(ris.dist(th0, r0+dr) - ris.dist(th0, r0))
    _,Vr,_ = V_components(ris, th0, r0+dr, th0, r0, None)
    print(f"  range  dr=2m:          V_r={Vr:.4f}  meas_var={np.var(psi):.4f}")

    # ---- (ii) Monte-Carlo: does the bound dR <= -log2((1-V/2)^2) hold? ----
    # additivity check: Var_m(psi_total) ?= V_a+V_r+V_q
    print("\nAdditivity of variances (combined error): measured Var vs V_a+V_r+V_q")
    for _ in range(4):
        th=np.deg2rad(rng.uniform(-40,40)); r=rng.uniform(10,40)
        dth=np.deg2rad(rng.normal(0,0.2)); dr=rng.normal(0,1.5); B=int(rng.choice([3,4,5]))
        th_hat,r_hat=th-dth,r-dr
        ph=quantize_phase(ris.k*ris.dist(th_hat,r_hat),B)
        psi=ris.k*ris.dist(th,r)-ph
        Va,Vr,Vq=V_components(ris,th,r,th_hat,r_hat,B)
        print(f"  meas_var={np.var(psi):.4f}   V_a+V_r+V_q={Va+Vr+Vq:.4f}")

    print("\nMonte-Carlo: tight bound dR <= -log2((1-(V_a+V_r+V_q)/2)^2)")
    viol = 0; N_mc = 5000; rows = []
    for _ in range(N_mc):
        th = np.deg2rad(rng.uniform(-50,50)); r = rng.uniform(8.0, 45.0)
        dth = np.deg2rad(rng.normal(0,0.2)); dr = rng.normal(0,1.5)
        B = int(rng.choice([2,3,4,5]))
        th_hat, r_hat = th-dth, r-dr
        g = exact_gain(ris, th, r, th_hat, r_hat, B)
        dR = np.log2((1+gamma)/(1+gamma*g))
        Va,Vr,Vq = V_components(ris, th, r, th_hat, r_hat, B)
        V = Va+Vr+Vq
        g_lb = (1-V/2)**2 if V < 2 else 1e-12
        dR_bound = -np.log2(max(g_lb,1e-12))
        if dR > dR_bound + 1e-9: viol += 1
        rows.append((dR, dR_bound, g, max(g_lb,0)))
    rows = np.array(rows)
    print(f"  trials={N_mc}  violations={viol}  ({100*viol/N_mc:.2f}%)")
    print(f"  mean actual dR={rows[:,0].mean():.3f}  mean bound={rows[:,1].mean():.3f} bit/s/Hz")
    print(f"  mean exact g={rows[:,2].mean():.4f}  mean g_lb={rows[:,3].mean():.4f}")
    print(f"  bound holds = {viol==0}")
