"""Predictor ablation: justify the Attention-LSTM architecture. Train variants on
the same data and compare held-out accuracy."""
import sys; sys.path.insert(0,".")
import numpy as np, torch, json
from mobility import build_dataset
from predictor import train_predictor

def acc(model, std, X, Y, vi):
    with torch.no_grad():
        p=std.inv_y(model(torch.tensor(std.tx(X[vi]))).numpy())
    t=Y[vi]
    return (np.rad2deg(np.sqrt(np.mean((p[:,0]-t[:,0])**2))),
            np.sqrt(np.mean((p[:,1]-t[:,1])**2)))

X,Y=build_dataset(n_traj=1200,n_steps=200,stride=6,obs_sig_th_deg=0.1,obs_sig_r_frac=0.01,seed=1)
variants={
 "Attention-LSTM (3L+attn)": dict(layers=3,use_attn=True),
 "LSTM only (3L, no attn)":   dict(layers=3,use_attn=False),
 "Shallow LSTM (1L+attn)":    dict(layers=1,use_attn=True),
}
rows={}
for name,kw in variants.items():
    print("training:",name)
    m,std,hist,(ti,vi)=train_predictor(X,Y,epochs=20,seed=1,model_kwargs=kw,log=lambda *a:None)
    rt,rr=acc(m,std,X,Y,vi); rows[name]=(float(rt),float(rr))
    print(f"   RMSE_theta={rt:.3f} deg  RMSE_r={rr:.3f} m")
json.dump(rows,open("ablation.json","w"),indent=2)
print("\n=== ablation ===");
for k,v in rows.items(): print(f"{k:28s} theta={v[0]:.3f} deg  r={v[1]:.3f} m")
print("saved ablation.json")
