"""
predictor.py — Attention-LSTM predictor for near-field (angle,range) forecasting.

Three stacked LSTM layers followed by multi-head self-attention over the hidden
sequence and a linear head that outputs (theta, range) H slots ahead. Mirrors the
far-field architecture of Abdulhussien & Ibrahim (Phys. Commun. 2026), adapted to
predict the two-dimensional near-field coordinate.
"""
import numpy as np
import torch, torch.nn as nn

class AttnLSTM(nn.Module):
    def __init__(self, in_dim=4, hidden=64, layers=3, heads=2, out_dim=2, drop=0.1,
                 use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=drop)
        if use_attn:
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, out_dim))
    def forward(self, x):
        h, _ = self.lstm(x)                    # (B,L,H)
        if self.use_attn:
            a, _ = self.attn(h, h, h)          # self-attention
            z = self.norm(h + a)[:, -1]        # residual, take last step
        else:
            z = h[:, -1]
        return self.head(z)

class Standardizer:
    """Per-feature mean/std for inputs and targets (fit on train only)."""
    def fit(self, X, Y):
        self.xm = X.reshape(-1, X.shape[-1]).mean(0); self.xs = X.reshape(-1, X.shape[-1]).std(0)+1e-6
        self.ym = Y.mean(0); self.ys = Y.std(0)+1e-6
        return self
    def tx(self, X): return (X - self.xm)/self.xs
    def ty(self, Y): return (Y - self.ym)/self.ys
    def inv_y(self, Yn): return Yn*self.ys + self.ym

def train_predictor(X, Y, epochs=25, bs=256, lr=1e-3, val_frac=0.15,
                    seed=0, device="cpu", log=print, model_kwargs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X); idx = np.random.permutation(n); nv = int(val_frac*n)
    vi, ti = idx[:nv], idx[nv:]
    std = Standardizer().fit(X[ti], Y[ti])
    Xt = torch.tensor(std.tx(X)); Yt = torch.tensor(std.ty(Y))
    model = AttnLSTM(**(model_kwargs or {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.MSELoss()
    tr_t = torch.tensor(ti); va_t = torch.tensor(vi)
    hist = []
    for ep in range(epochs):
        model.train(); perm = tr_t[torch.randperm(len(tr_t))]
        tl = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xt[b].to(device))
            loss = lossf(out, Yt[b].to(device))
            loss.backward(); opt.step(); tl += loss.item()*len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vout = model(Xt[va_t].to(device))
            vpred = std.inv_y(vout.cpu().numpy()); vtrue = Y[vi]
        rmse_th = np.rad2deg(np.sqrt(np.mean((vpred[:,0]-vtrue[:,0])**2)))
        rmse_r = np.sqrt(np.mean((vpred[:,1]-vtrue[:,1])**2))
        hist.append((ep, tl/len(tr_t), rmse_th, rmse_r))
        if ep % 5 == 0 or ep == epochs-1:
            log(f"  epoch {ep:2d}  train_mse={tl/len(tr_t):.4f}  "
                f"val RMSE_theta={rmse_th:.3f} deg  RMSE_r={rmse_r:.3f} m")
    return model, std, hist, (ti, vi)

def baseline_errors(X, Y, std=None):
    """Last-value (reactive) and linear (constant-velocity) extrapolation errors.
    X[:, -1] = [theta,range,dtheta,drange] at the most recent observed slot."""
    th_last, r_last = X[:,-1,0], X[:,-1,1]
    dth, dr = X[:,-1,2], X[:,-1,3]
    H = 40
    # last value
    lv_th = np.rad2deg(np.sqrt(np.mean((th_last - Y[:,0])**2)))
    lv_r = np.sqrt(np.mean((r_last - Y[:,1])**2))
    # linear extrapolation H steps at last per-slot delta
    lin_th = np.rad2deg(np.sqrt(np.mean((th_last + H*dth - Y[:,0])**2)))
    lin_r = np.sqrt(np.mean((r_last + H*dr - Y[:,1])**2))
    return dict(lastval=(lv_th,lv_r), linear=(lin_th,lin_r))
