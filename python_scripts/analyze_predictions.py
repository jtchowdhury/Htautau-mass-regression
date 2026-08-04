"""
analyze_predictions.py
----------------------
Analysis for a salt `test` predictions h5. Works on the flat-mass validation
set (per-jet truth Higgs mass) and on the SM samples (fixed truth mass, e.g.
125 GeV) via the --mh flag.

Produces, into an analysis_plots/ folder next to the input file:
  1. mass_<label>.png        reco vs regressed mass distributions
                             (+ truth Higgs for flat-mass; vertical line at the
                              expected mass) -> the "two peaks" plot for SM
  2. resolution_<label>.png  relative response  (m / mH - 1),  reco vs regressed
  3. residual_<label>.png    absolute residual  (m - mH) [GeV], reco vs regressed

Usage:
    # flat-mass validation (per-jet truth from the file):
    python analyze_predictions.py <val_predictions.h5> --label "flat-mass BSM"

    # SM sample (fixed truth Higgs mass 125 GeV):
    python analyze_predictions.py <sm_predictions.h5> --mh 125 --label "SM HH bbtautau"

Fields:
    reco      : jets['mass']                                      (MeV)
    regressed : jets['htautau_mass_regression_GhostHBosonsMass']  (MeV)
    truth     : jets['GhostHBosonsMass']  (MeV)  [flat-mass only; use --mh for SM]
"""
import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUTH_FIELD = "GhostHBosonsMass"
PRED_FIELD  = "htautau_mass_regression_GhostHBosonsMass"

C_TRUTH = "#1D9E75"   # teal
C_RECO  = "#5B4FCF"   # purple
C_REG   = "#D85A30"   # coral

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 12})


def load(path, mh):
    with h5py.File(path, "r") as f:
        j = f["jets"][:]
    reco = j["mass"].astype(np.float64) / 1e3           # GeV
    pred = j[PRED_FIELD].astype(np.float64) / 1e3
    good = np.isfinite(pred) & np.isfinite(reco) & (reco > 0)
    if mh is not None:                                  # SM: fixed truth mass
        true = np.full(reco.shape, float(mh))
    else:                                               # flat-mass: per-jet truth
        true = j[TRUTH_FIELD].astype(np.float64) / 1e3
        good &= (true > 0) & np.isfinite(true)
    return true[good], pred[good], reco[good]


def med_iqr(x):
    return np.median(x), np.percentile(x, 75) - np.percentile(x, 25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="salt test predictions .h5")
    ap.add_argument("--mh", type=float, default=None,
                    help="fixed truth Higgs mass in GeV (SM samples). "
                         "Omit to use per-jet GhostHBosonsMass (flat-mass).")
    ap.add_argument("--label", default="sample", help="title / filename tag")
    ap.add_argument("--ref", type=float, default=None,
                    help="vertical reference line on the mass plot [GeV] "
                         "(defaults to --mh if given)")
    ap.add_argument("--mcut", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="keep only jets with LO <= truth Higgs mass [GeV] <= HI")
    args = ap.parse_args()

    true, pred, reco = load(args.path, args.mh)
    tag = args.label.replace(" ", "_")

    if args.mcut:                       # cut on truth Higgs mass
        lo, hi = args.mcut
        keep = (true >= lo) & (true <= hi)
        true, pred, reco = true[keep], pred[keep], reco[keep]
        tag += f"_m{lo:.0f}-{hi:.0f}"
        args.label += f" [{lo:.0f}-{hi:.0f} GeV]"

    ref = args.ref if args.ref is not None else args.mh
    per_jet_truth = args.mh is None

    resp_reco = reco / true - 1
    resp_reg  = pred / true - 1
    res_reco  = reco - true      # GeV
    res_reg   = pred - true

    mr, ir = med_iqr(resp_reco); mg, ig = med_iqr(resp_reg)
    dr, jr = med_iqr(res_reco);  dg, jg = med_iqr(res_reg)
    print(f"[{args.label}]  N = {true.size:,}")
    print(f"  reco      mean {reco.mean():.1f} GeV | response med {mr:+.3f} IQR {ir:.3f} | residual med {dr:+.1f} GeV")
    print(f"  regressed mean {pred.mean():.1f} GeV | response med {mg:+.3f} IQR {ig:.3f} | residual med {dg:+.1f} GeV")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(base_dir, "plots", "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    # ---- 1. mass distributions (the "two peaks" view) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if per_jet_truth:
        ax.hist(true, bins=80, range=(40, 250), histtype="step", density=True,
                color=C_TRUTH, lw=2, ls="-", label="truth Higgs")
    ax.hist(reco, bins=80, range=(40, 250), histtype="step", density=True,
            color=C_RECO, lw=2, ls=":", label="reco")
    ax.hist(pred, bins=80, range=(40, 250), histtype="step", density=True,
            color=C_REG, lw=2, ls="-", label="regressed")
    if ref is not None:
        ax.axvline(ref, color="black", ls="--", lw=1, alpha=0.6,
                   label=f"$m_H$ = {ref:g} GeV")
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title(f"{args.label}: reco vs regressed mass")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"mass_{tag}.png"), dpi=150); plt.close()

    # ---- 2. relative response (m / mH - 1) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(resp_reco, bins=100, range=(-1, 1), histtype="step", density=True,
            color=C_RECO, lw=2, label=f"reco / truth        (median={mr:.2f}, IQR={ir:.2f})")
    ax.hist(resp_reg, bins=100, range=(-1, 1), histtype="step", density=True,
            color=C_REG, lw=2, label=f"regressed / truth  (median={mg:.2f}, IQR={ig:.2f})")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="perfect response")
    ax.set_xlabel(r"$(m / m_H) - 1$")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Relative mass resolution: reco vs regressed ({args.label})")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"resolution_{tag}.png"), dpi=150); plt.close()

    # ---- 3. absolute residual (m - mH) [GeV] ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(res_reco, bins=100, range=(-100, 100), histtype="step", density=True,
            color=C_RECO, lw=2, label=f"reco - $m_H$        (median={dr:.1f}, IQR={jr:.1f} GeV)")
    ax.hist(res_reg, bins=100, range=(-100, 100), histtype="step", density=True,
            color=C_REG, lw=2, label=f"regressed - $m_H$  (median={dg:.1f}, IQR={jg:.1f} GeV)")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="perfect")
    ax.set_xlabel(r"$m - m_H$ [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title(f"Absolute mass residual: reco vs regressed ({args.label})")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"residual_{tag}.png"), dpi=150); plt.close()

    print(f"plots -> {outdir}/[mass|resolution|residual]_{tag}.png")


if __name__ == "__main__":
    main()
