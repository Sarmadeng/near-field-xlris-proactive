"""
mobility.py — Gauss-Markov near-field mobility trajectories for XL-RIS.

User moves in the 2-D plane in front of the surface (at the origin). Velocity
follows a Gauss-Markov process (memory alpha), position integrates it. We keep
the user inside the near-field service region (range in [r_min,r_max], angle in
[-theta_max,theta_max]) by specular reflection at the boundaries. Observables
are the polar coordinates seen by the surface: angle theta[n] and range r[n].
"""
import numpy as np

def gauss_markov_trajectory(n_steps, T=0.5e-3, alpha=0.9, speed=15.0,
                            sigma_v=4.0, r_min=8.0, r_max=45.0,
                            theta_max_deg=60.0, rng=None):
    rng = rng or np.random.default_rng()
    th_max = np.deg2rad(theta_max_deg)
    # start pos
    r0 = rng.uniform(r_min+2, r_max-2); th0 = rng.uniform(-th_max*0.8, th_max*0.8)
    p = np.array([r0*np.sin(th0), r0*np.cos(th0)])      # x (cross), y (range axis)
    # mean velocity: random heading at target speed
    head = rng.uniform(0, 2*np.pi)
    v_mean = speed*np.array([np.cos(head), np.sin(head)])
    v = v_mean.copy()
    out = np.zeros((n_steps, 2))
    for n in range(n_steps):
        w = rng.standard_normal(2)
        v = alpha*v + (1-alpha)*v_mean + np.sqrt(1-alpha**2)*sigma_v*w
        p = p + v*T
        rr = np.hypot(p[0], p[1]); th = np.arctan2(p[0], p[1])
        # reflect at boundaries
        if rr < r_min or rr > r_max:
            # reflect radial velocity
            u = p/max(rr,1e-9); v = v - 2*np.dot(v,u)*u
            rr = np.clip(rr, r_min, r_max); p = rr*np.array([np.sin(th), np.cos(th)])
        if abs(th) > th_max:
            v[0] = -v[0]; th = np.clip(th, -th_max, th_max)
            p = rr*np.array([np.sin(th), np.cos(th)])
        out[n] = [th, rr]
    return out    # (n_steps,2): [theta(rad), range(m)]

def build_dataset(n_traj=2500, n_steps=220, L=64, H=40, speeds=(5,10,15,20,25,30),
                  stride=6, obs_sig_th_deg=0.5, obs_sig_r_frac=0.02, seed=0):
    """Windowed samples: X = past L NOISY observations (theta,range,dtheta,drange),
    Y = TRUE (theta,range) at +H. Angle/range observations carry estimation noise
    (pilot-based estimation), so the predictor must filter as well as forecast;
    this is where a learned model beats naive linear extrapolation of noisy samples.
    `stride` subsamples the sliding windows to keep the dataset memory-bounded."""
    rng = np.random.default_rng(seed)
    sth = np.deg2rad(obs_sig_th_deg)
    X, Y = [], []
    for t in range(n_traj):
        sp = rng.choice(speeds)
        tr = gauss_markov_trajectory(n_steps, speed=float(sp), rng=rng)
        th, r = tr[:,0], tr[:,1]                         # TRUE states
        th_o = th + sth*rng.standard_normal(n_steps)     # noisy observations
        r_o  = r * (1.0 + obs_sig_r_frac*rng.standard_normal(n_steps))
        dth = np.diff(th_o, prepend=th_o[0]); dr = np.diff(r_o, prepend=r_o[0])
        feat = np.stack([th_o, r_o, dth, dr], axis=1)    # (n_steps,4) NOISY inputs
        for n in range(L, n_steps-H, stride):
            X.append(feat[n-L:n]); Y.append([th[n+H], r[n+H]])   # clean targets
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    return X, Y

if __name__ == "__main__":
    X, Y = build_dataset(n_traj=50, n_steps=220)
    print("X", X.shape, "Y", Y.shape)
    print("theta range [deg]", np.rad2deg([Y[:,0].min(), Y[:,0].max()]))
    print("range [m]", [Y[:,1].min(), Y[:,1].max()])
