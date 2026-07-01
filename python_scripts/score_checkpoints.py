"""
score_checkpoints.py
--------------------
Rank saved checkpoints by REAL regression performance.

The `val_loss=0.00000` in the checkpoint filenames is a broken placeholder
(the config monitors a classification metric that doesn't exist in this
regression run). To find the genuinely best epoch, run `salt test` on each
checkpoint first (see job_scripts/run_score.sh) — that writes one
`epoch=XXX-...__test_htautau_val.h5` per checkpoint next to the .ckpt — then
this script reads them all and ranks by RMSE / median mass resolution.

Usage:
    python score_checkpoints.py <ckpts_dir>
"""
import os
import re
import sys
import glob
import numpy as np
import h5py

TRUTH_FIELD = "GhostHBosonsMass"
PRED_FIELD  = "htautau_mass_regression_GhostHBosonsMass"


def score(path):
    with h5py.File(path, "r") as f:
        j = f["jets"][:]
    true = j[TRUTH_FIELD].astype(np.float64) / 1e3   # GeV
    pred = j[PRED_FIELD].astype(np.float64) / 1e3
    m = (true > 0) & np.isfinite(true) & np.isfinite(pred)
    true, pred = true[m], pred[m]
    resp = pred / true - 1
    return dict(
        n=true.size,
        rmse=float(np.sqrt(np.mean((pred - true) ** 2))),
        mae=float(np.mean(np.abs(pred - true))),
        bias=float(np.median(resp)),
        iqr=float(np.percentile(resp, 75) - np.percentile(resp, 25)),
    )


def main():
    ckdir = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(ckdir, "*__test_*.h5")))
    if not files:
        print(f"No prediction files (*__test_*.h5) found in {ckdir}")
        print("Run job_scripts/run_score.sh first to generate them.")
        return

    rows = []
    for fp in files:
        m = re.search(r"epoch=(\d+)", os.path.basename(fp))
        ep = int(m.group(1)) if m else -1
        s = score(fp)
        s["epoch"] = ep
        rows.append(s)

    rows.sort(key=lambda r: r["rmse"])
    print(f"{'epoch':>5} {'N':>8} {'RMSE[GeV]':>10} {'MAE[GeV]':>9} {'bias':>8} {'IQR':>7}")
    print("-" * 52)
    for r in rows:
        print(f"{r['epoch']:>5} {r['n']:>8} {r['rmse']:>10.2f} "
              f"{r['mae']:>9.2f} {r['bias']:>+8.3f} {r['iqr']:>7.3f}")
    best = rows[0]
    print(f"\nBest by RMSE: epoch {best['epoch']}  "
          f"(RMSE {best['rmse']:.2f} GeV, bias {best['bias']:+.3f}, IQR {best['iqr']:.3f})")


if __name__ == "__main__":
    main()
