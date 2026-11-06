#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "[1] Theorem 1";           python3 verify_thm1.py
echo "[2] Theorem 2";           python3 verify_thm2.py
echo "[3] UPA corollary";       python3 nf_upa.py
echo "[4] Train predictor";     python3 train_run.py
echo "[5] End-to-end SE";       python3 e2e_trained.py
echo "[6] Ablation";            python3 ablation.py
echo "[7] Horizon";             python3 horizon.py
echo "[8] Figures";             python3 fig_thm1.py; python3 fig_upa.py; python3 fig_e2e_trained.py; \
                                python3 fig_sweeps.py; python3 fig_horizon.py; \
                                python3 scenario_results.py; python3 fig_multiuser.py
echo "DONE — 7 figures in ../figs/"
