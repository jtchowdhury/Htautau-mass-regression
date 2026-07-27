"""
plot_loss_vs_epoch.py
---------------------
Reconstruct the validation-loss-vs-epoch curve directly from the per-checkpoint
salt `test` prediction files (epoch=XXX-...__test_htautau_val.h5).

The quantity salt minimises is MSE on the *standardised* target. With target
standardisation (x - mean)/std, the mean cancels in the difference, so:

    val_loss(epoch) = mean( (pred - true)^2 ) / std^2

i.e. the physical MSE divided by the target variance. std is the norm_params
value from the config. RMSE in GeV is also reported for interpretability.

Usage:
    python plot_loss_vs_epoch.py <ckpts_dir>
"""
import os
import re
import sys
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUTH_FIELD = "GhostHBosonsMass"
PRED_FIELD  = "htautau_mass_regression_GhostHBosonsMass"
STD_MEV     = 36738.14   # norm_params std from htautau_regression.yaml (MeV)


def loss_for(path):
    with h5py.File(path, "r") as f:
        j = f["jets"][:]
    true = j[TRUTH_FIELD].astype(np.float64)
    pred = j[PRED_FIELD].astype(np.float64)
    m = (true > 0) & np.isfinite(true) & np.isfinite(pred)
    true, pred = true[m], pred[m]
    mse = float(np.mean((pred - true) ** 2))     # MeV^2
    return mse / STD_MEV ** 2, np.sqrt(mse) / 1e3, true.size   # val_loss, RMSE[GeV], N


def main():
    ckdir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(ckdir, "*__test_*.h5")))
    if not files:
        print(f"No prediction files (*__test_*.h5) in {ckdir}. Run run_score first.")
        return

    rows = []
    for fp in files:
        m = re.search(r"epoch=(\d+)", os.path.basename(fp))
        if not m:
            continue
        ep = int(m.group(1))
        vl, rmse, n = loss_for(fp)
        rows.append((ep, vl, rmse, n))
    rows.sort()
    eps   = np.array([r[0] for r in rows])
    vloss = np.array([r[1] for r in rows])
    rmse  = np.array([r[2] for r in rows])
    n_jets = rows[0][3]

    print(f"{'epoch':>5} {'val_loss':>10} {'RMSE[GeV]':>10}")
    for ep, vl, rm, _ in rows:
        print(f"{ep:>5} {vl:>10.4f} {rm:>10.2f}")
    best = int(eps[np.argmin(vloss)])
    print(f"\nMin val_loss at epoch {best}  "
          f"(val_loss {vloss.min():.4f}, RMSE {rmse[np.argmin(vloss)]:.2f} GeV)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, vloss, "o-", lw=2, color="#1D9E75")
    ax.axvline(best, color="k", ls="--", lw=1, alpha=0.6,
               label=f"min at epoch {best}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"validation loss  (standardised MSE)")
    ax.set_title(f"Validation loss vs epoch  (N = {n_jets:,} val jets)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(ckdir, "val_loss_vs_epoch.png")
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
