"""
analyze_predictions.py
----------------------
Analysis for a salt `test` predictions h5 (flat-mass sample, 802168).
Produces two plots in the style of explore_htautau.py:

  1. mass_regressed.png   truth Higgs / reco / REGRESSED mass distributions
  2. resolution.png       response (m/m_truthHiggs - 1): reco vs regressed

Usage:
    python analyze_predictions.py <predictions.h5>

Fields (from salt test output):
    truth     : jets['GhostHBosonsMass']                          (MeV)
    regressed : jets['htautau_mass_regression_GhostHBosonsMass']  (MeV)
    reco      : jets['mass']                                      (MeV)

Note: truth-jet mass is not in this file (not carried through
prepare_htautau.py). Add 'R10TruthLabel_R22v1_TruthJetMass' to the jet
output there and regenerate if you want it on the mass plot.
"""
import os
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUTH_FIELD = "GhostHBosonsMass"
PRED_FIELD  = "htautau_mass_regression_GhostHBosonsMass"

# colours matching explore_htautau.py
C_TRUTH = "#1D9E75"   # teal
C_RECO  = "#5B4FCF"   # purple
C_REG   = "#D85A30"   # coral (the regressed / new curve)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})


def load(path):
    with h5py.File(path, "r") as f:
        j = f["jets"][:]
    true = j[TRUTH_FIELD].astype(np.float64) / 1e3   # GeV
    pred = j[PRED_FIELD].astype(np.float64) / 1e3
    reco = j["mass"].astype(np.float64) / 1e3
    m = (true > 0) & np.isfinite(true) & np.isfinite(pred) & np.isfinite(reco)
    return true[m], pred[m], reco[m]


def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_predictions.py <predictions.h5>")
        return
    path = sys.argv[1]
    true, pred, reco = load(path)

    resp_reco = reco / true - 1
    resp_reg  = pred / true - 1

    def stats(r):
        return np.median(r), np.percentile(r, 75) - np.percentile(r, 25)

    med_reco, iqr_reco = stats(resp_reco)
    med_reg,  iqr_reg  = stats(resp_reg)

    print(f"N = {true.size:,}")
    print(f"true mean {true.mean():.1f} GeV | reco mean {reco.mean():.1f} GeV | regressed mean {pred.mean():.1f} GeV")
    print(f"RMSE  reco {np.sqrt(np.mean((reco-true)**2)):.2f} GeV | regressed {np.sqrt(np.mean((pred-true)**2)):.2f} GeV")
    print(f"reco       response median {med_reco:+.3f}  IQR {iqr_reco:.3f}")
    print(f"regressed  response median {med_reg:+.3f}  IQR {iqr_reg:.3f}")

    outdir = os.path.join(os.path.dirname(os.path.abspath(path)), "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    # ---- Plot 1: mass distributions (truth Higgs / reco / regressed) ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(true, bins=80, range=(40, 250), histtype="step", density=True,
            color=C_TRUTH, lw=2, ls="-",  label="truth Higgs")
    ax.hist(reco, bins=80, range=(40, 250), histtype="step", density=True,
            color=C_RECO,  lw=2, ls=":",  label="reco")
    ax.hist(pred, bins=80, range=(40, 250), histtype="step", density=True,
            color=C_REG,   lw=2, ls="-",  label="regressed")
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title("Flat-mass BSM (802168): regressed mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mass_regressed.png"), dpi=150)
    plt.close()

    # ---- Plot 2: mass resolution (reco vs regressed) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(resp_reco, bins=100, range=(-1, 1), histtype="step", density=True,
            color=C_RECO, lw=2,
            label=f"reco / truth        (median={med_reco:.2f}, IQR={iqr_reco:.2f})")
    ax.hist(resp_reg, bins=100, range=(-1, 1), histtype="step", density=True,
            color=C_REG, lw=2,
            label=f"regressed / truth  (median={med_reg:.2f}, IQR={iqr_reg:.2f})")
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="perfect response")
    ax.set_xlabel(r"$(m / m_\mathrm{truth\,Higgs}) - 1$")
    ax.set_ylabel("Normalised")
    ax.set_title("Mass resolution: reco vs regressed (flat-mass BSM)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "resolution.png"), dpi=150)
    plt.close()

    print(f"\nplots -> {outdir}/mass_regressed.png, {outdir}/resolution.png")


if __name__ == "__main__":
    main()
