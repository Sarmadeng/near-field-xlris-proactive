# Proactive Near-Field Beam-Focusing for XL-RIS–Aided 6G Links

Reproducible code for the paper

> **Proactive Near-Field Beam-Focusing for XL-RIS–Aided 6G Links: A
> Predict-then-Compute Framework with Provable Focusing Guarantees**
> S. K. Ibrahim, H. S. Saeed, A. A. Sharba.

A proactive **predict-then-compute** framework for extremely large RIS (XL-RIS)
operating in the radiating **near field**: an Attention-LSTM forecasts the user's
(angle, range) one reconfiguration interval ahead and the focusing phases are
computed in **closed form** — no beam training, no reinforcement learning at
inference. The repository reproduces every theorem check, the trained predictor,
and every figure.

## Theorems (all numerically verified)
1. **Near-field focusing gain & depth of focus** — closed-form Fresnel gain
   matching the exact element sum to `< 8e-4`; exact 3-dB depth-of-focus and the
   *effective focusing range* `1/beta`.  (`verify_thm1.py`)
2. **Rate-loss bound** under angle/range prediction error and B-bit phase
   quantisation; additive decomposition with the key `D^4` range sensitivity.
   Zero bound violations over 6000 trials. (`verify_thm2.py`)
3. **Complexity** `O(N)` closed-form vs `O(N·N_A·S_R)` near-field beam training.

## Trained predictor (real)
Attention-LSTM (3 LSTM layers + multi-head self-attention) trained on
Gauss–Markov near-field trajectories with realistic estimation noise. Held-out
accuracy (20 ms / 40-slot horizon):

| Predictor | angle RMSE | range RMSE |
|---|---|---|
| Attention-LSTM (proposed) | **0.30°** | **0.31 m (1.37%)** |
| Last value (reactive) | 0.81° | 0.40 m |
| Linear (constant velocity) | 5.71° | 15.63 m |

## Layout
```
sim/nf_core.py          near-field XL-RIS array model + Fresnel focusing gain
sim/mobility.py         Gauss–Markov near-field trajectory + dataset builder
sim/predictor.py        Attention-LSTM model + training loop
sim/train_run.py        build dataset, train predictor, report accuracy
sim/e2e_trained.py      end-to-end spectral efficiency with the trained model
sim/verify_thm1.py      Theorem 1 verification
sim/verify_thm2.py      Theorem 2 verification
sim/fig_thm1.py         Figure 1 (focusing gain / depth of focus)
sim/fig_e2e_trained.py  Figure 2 (mobility robustness / near-field advantage)
sim/nf_upa.py           planar (UPA) near-field focusing + separability corollary
sim/fig_upa.py          UPA figure (separability + 2-D focal spot)
sim/fig_sweeps.py       SE vs quantisation bits B and surface size N
sim/ablation.py         predictor architecture ablation
sim/horizon.py          prediction accuracy vs forecast horizon (latency)
sim/scenario_results.py SE vs SNR and SE vs latency
sim/fig_multiuser.py    multi-user range-domain separation
sim/run_all.sh          run everything end to end
```

## Reproduce
```bash
pip install -r requirements.txt
cd sim
python verify_thm1.py      # Theorem 1
python verify_thm2.py      # Theorem 2
python train_run.py        # train predictor (~3 min CPU)
python e2e_trained.py      # end-to-end SE
python fig_thm1.py         # Figure 1
python fig_e2e_trained.py  # Figure 2
# or simply:
bash run_all.sh
```

Random seeds are fixed; results are reported as mean ± std. All numbers in the
paper are produced by this code — no value is hand-entered.

## License
MIT (see `LICENSE`).
