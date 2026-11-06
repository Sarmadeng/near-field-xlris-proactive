"""Prediction accuracy vs forecast horizon H (reconfiguration latency).
Trains the predictor at several horizons and records held-out RMSE, quantifying
how far ahead the Attention-LSTM can forecast before accuracy degrades."""
import sys; sys.path.insert(0,".")
import numpy as np, torch, json
from mobility import build_dataset
from predictor import train_predictor

rows=[]
for H in [10,20,40,60,80]:
    X,Y=build_dataset(n_traj=1000,n_steps=220,L=64,H=H,stride=6,
                      obs_sig_th_deg=0.1,obs_sig_r_frac=0.01,seed=1)
    m,std,hist,(ti,vi)=train_predictor(X,Y,epochs=18,seed=1,log=lambda *a:None)
    with torch.no_grad():
        p=std.inv_y(m(torch.tensor(std.tx(X[vi]))).numpy())
    t=Y[vi]
    rt=np.rad2deg(np.sqrt(np.mean((p[:,0]-t[:,0])**2)))
    rr=np.sqrt(np.mean((p[:,1]-t[:,1])**2))
    ms=0.5*H  # latency in ms at 0.5 ms/slot
    rows.append((H,ms,float(rt),float(rr)))
    print(f"H={H:3d} ({ms:4.0f} ms):  RMSE_theta={rt:.3f} deg  RMSE_r={rr:.3f} m")
json.dump(rows,open("horizon.json","w"),indent=2)
print("saved horizon.json")
