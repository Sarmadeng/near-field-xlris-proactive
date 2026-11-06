"""Figure: prediction RMSE vs forecast horizon (reconfiguration latency)."""
import sys; sys.path.insert(0,".")
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=json.load(open("horizon.json"))
H=[r[0] for r in rows]; ms=[r[1] for r in rows]; rt=[r[2] for r in rows]; rr=[r[3] for r in rows]
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3})
fig,ax=plt.subplots(figsize=(5.2,3.6))
ax.plot(ms,rt,"o-",color="tab:red",lw=1.8,ms=6,label="angle RMSE [deg]")
bw=np.rad2deg(0.0107/1.370)
ax.axhline(bw,color="0.5",ls=":",lw=1.2,label=f"beamwidth $\\lambda/D$={bw:.2f} deg")
ax.set_xlabel("reconfiguration latency $\\tau_{lat}$ [ms]")
ax.set_ylabel("angle prediction RMSE [deg]")
ax2=ax.twinx(); ax2.plot(ms,rr,"s--",color="tab:blue",lw=1.5,ms=5,label="range RMSE [m]")
ax2.set_ylabel("range prediction RMSE [m]",color="tab:blue")
ax.set_title("Prediction accuracy vs forecast horizon")
ax.legend(fontsize=8,loc="upper left")
fig.tight_layout(); fig.savefig("../figs/fig_horizon.pdf")
print("saved ../figs/fig_horizon.pdf"); print("rows:",rows)
