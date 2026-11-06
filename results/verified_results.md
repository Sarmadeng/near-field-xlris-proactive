# Verified results — Near-field proactive XL-RIS focusing (Physical Communication)

System: ULA XL-RIS, N=256, fc=28 GHz, lambda=10.71 mm, d=lambda/2,
aperture D=1.370 m, Rayleigh distance 2D^2/lambda = 350.8 m, Fresnel 7.75 m.

## Theorem 1 — near-field focusing gain & depth of focus  (VERIFIED <1e-3)
Broadside range focusing gain (Fresnel closed form) matches exact element sum to
max |err| = 8e-4 over a 5–140 m range sweep:
    g(r) = (C(a)^2 + S(a)^2)/a^2,   a = (D/2) sqrt( 2 |1/r - 1/r0| / lambda ),
    C,S Fresnel integrals (pi/2 convention).
3-dB depth of focus (matches measured interval EXACTLY):
    r in [ 1/(1/r0 + beta), 1/(1/r0 - beta) ],  beta = 2 a_L^2 lambda / D^2,
    a_L = 1.616 (50% crossing of g). Far edge finite only if r0 < 1/beta.
Effective focusing range 1/beta = 50.5 m = 0.144 * Rayleigh.
    r0=10 -> DoF [8.3,12.5] m; r0=20 -> [14.3,33.1]; r0=40 -> [22.3,192.9];
    r0=80 -> [30.9, inf] (beyond 1/beta, range-unselective).

## Theorem 2 — rate-loss bound under prediction error + B-bit quantisation (VERIFIED)
Rigorous: dR = log2((1+gamma)/(1+gamma*g)) <= -log2((1-V/2)^2), V<2,
    V = Var_m(psi_m) element-phase-error variance.  0 violations / 6000 trials.
Closed-form decomposition (median error ~2% in small-error regime):
    V ~= V_a + V_r + V_q,
    V_a = k^2 cos^2(theta) dtheta^2 * (D^2/12)      [angular pred. error]
    V_r = (k^2/4) (1/r - 1/rhat)^2 * (D^4/180)       [range pred. error]
    V_q = pi^2 / (3 * 4^B)                            [B-bit phase quantisation]
Single-source checks: V_a 1.4305 vs meas 1.4309; V_r 0.0349 vs meas 0.0328.
Additivity confirmed: Var(psi_total) ~= V_a+V_r+V_q (e.g. 0.3076 vs 0.3114).
KEY INSIGHT: range error scales with D^4 vs D^2 for angle => near-field range
focusing is far more sensitive to range-prediction accuracy at large aperture.

## Trained Attention-LSTM predictor (REAL training) + end-to-end SE
Dataset: Gauss-Markov near-field trajectories (alpha=0.9, speeds 5-30 m/s,
r in [8,45] m, |theta|<=60 deg), obs noise 0.1 deg (angle), 1% (range).
Window L=64, horizon H=40 slots (20 ms @ 0.5 ms), 16320 train / 2880 val windows,
20 epochs. Held-out accuracy:
    Attention-LSTM : RMSE_theta=0.303 deg, RMSE_r=0.312 m (1.37%)
    Last value     : RMSE_theta=0.806 deg, RMSE_r=0.402 m
    Linear (const-v): RMSE_theta=5.71 deg, RMSE_r=15.63 m
End-to-end effective SE (N=256, B=4, rhoN=20 dB), genie=6.64 bit/s/Hz:
    proposed flat ~5.2 (~78% genie); reactive 6.0 -> 2.0 (collapses);
    far-field ~4.0; crossover at v~10 m/s; proposed/reactive=2.40x at 30 m/s.
Near-field advantage (Thm 1) concentrated at r0 < 1/beta = 50.5 m.
