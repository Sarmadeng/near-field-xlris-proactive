"""Build dataset, train the Attention-LSTM predictor, report real accuracy vs
baselines, and save the trained model + validation predictions for the e2e sim."""
import sys; sys.path.insert(0,".")
import numpy as np, torch, json, time
from mobility import build_dataset
from predictor import train_predictor, baseline_errors, AttnLSTM, Standardizer

def main():
    t0=time.time()
    print("Building dataset (Gauss-Markov near-field trajectories)...")
    X, Y = build_dataset(n_traj=1200, n_steps=200, L=64, H=40, stride=6,
                         obs_sig_th_deg=0.1, obs_sig_r_frac=0.01, seed=1)
    print(f"  samples: {X.shape[0]}  window L=64  horizon H=40 slots (20 ms @0.5ms)")
    print("Training Attention-LSTM (3 LSTM layers + 2-head self-attention)...")
    model, std, hist, (ti,vi) = train_predictor(X, Y, epochs=20, seed=1)
    # final validation accuracy
    with torch.no_grad():
        Xt=torch.tensor(std.tx(X[vi])); pred=std.inv_y(model(Xt).numpy())
    true=Y[vi]
    rmse_th=np.rad2deg(np.sqrt(np.mean((pred[:,0]-true[:,0])**2)))
    rmse_r=np.sqrt(np.mean((pred[:,1]-true[:,1])**2))
    r_rel=100*np.sqrt(np.mean(((pred[:,1]-true[:,1])/true[:,1])**2))
    bl=baseline_errors(X[vi], Y[vi])
    print("\n=== REAL prediction accuracy (held-out validation) ===")
    print(f"  Attention-LSTM : RMSE_theta={rmse_th:.3f} deg   RMSE_r={rmse_r:.3f} m ({r_rel:.2f}%)")
    print(f"  Linear (const-v): RMSE_theta={bl['linear'][0]:.3f} deg   RMSE_r={bl['linear'][1]:.3f} m")
    print(f"  Last value      : RMSE_theta={bl['lastval'][0]:.3f} deg   RMSE_r={bl['lastval'][1]:.3f} m")
    # save model + std + a sample of (pred_error) for e2e
    torch.save(model.state_dict(), "predictor_model.pt")
    np.savez("predictor_stats.npz",
             xm=std.xm, xs=std.xs, ym=std.ym, ys=std.ys,
             err_th=(pred[:,0]-true[:,0]), err_r=(pred[:,1]-true[:,1]),
             true_r=true[:,1])
    res=dict(rmse_theta_deg=float(rmse_th), rmse_r_m=float(rmse_r), rmse_r_pct=float(r_rel),
             lin_theta=float(bl['linear'][0]), lin_r=float(bl['linear'][1]),
             lv_theta=float(bl['lastval'][0]), lv_r=float(bl['lastval'][1]),
             n_val=int(len(vi)), n_train=int(len(ti)))
    json.dump(res, open("predictor_accuracy.json","w"), indent=2)
    print(f"\nsaved predictor_model.pt, predictor_stats.npz, predictor_accuracy.json")
    print(f"elapsed {time.time()-t0:.1f}s")

if __name__=="__main__":
    main()
