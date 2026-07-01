"""
analyze_predictions.py
----------------------
Analysis starter for a salt `test` predictions h5. Evaluates the H->tautau
Higgs-mass regression and reproduces the resolution views from Notes.txt.

Usage:
    python analyze_predictions.py <predictions.h5>

Fields (from salt test output):
    truth : jets['GhostHBosonsMass']                          (MeV)
    pred  : jets['htautau_mass_regression_GhostHBosonsMass']  (MeV)
    reco  : jets['mass']                                      (MeV)
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


def load(path):
    with h5py.File(path, "r") as f:
        j = f["jets"][:]
    true = j[TRUTH_FIELD].astype(np.float64) / 1e3   # GeV
    pred = j[PRED_FIELD].astype(np.float64) / 1e3
    reco = j["mass"].astype(np.float64) / 1e3
    pt   = j["pt"].astype(np.float64) / 1e3
    m = (true > 0) & np.isfinite(true) & np.isfinite(pred)
    return true[m], pred[m], reco[m], pt[m]


def profile(x, y, bins):
    cx, med, lo, hi = [], [], [], []
    for i in range(len(bins) - 1):
        s = (x >= bins[i]) & (x < bins[i + 1])
        if s.sum() < 20:
            continue
        q25, q50, q75 = np.percentile(y[s], [25, 50, 75])
        cx.append((bins[i] + bins[i + 1]) / 2)
        med.append(q50)
        lo.append(q50 - q25)
        hi.append(q75 - q50)
    return (np.array(cx), np.array(med), np.array(lo), np.array(hi))


def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_predictions.py <predictions.h5>")
        return
    path = sys.argv[1]
    true, pred, reco, pt = load(path)
    resp = pred / true - 1

    print(f"N = {true.size:,}")
    print(f"true mean {true.mean():.1f} GeV | pred mean {pred.mean():.1f} GeV | reco mean {reco.mean():.1f} GeV")
    print(f"RMSE {np.sqrt(np.mean((pred - true) ** 2)):.2f} GeV | MAE {np.mean(np.abs(pred - true)):.2f} GeV")
    print(f"median response {np.median(resp):+.3f} | IQR {np.percentile(resp, 75) - np.percentile(resp, 25):.3f}")

    outdir = os.path.join(os.path.dirname(os.path.abspath(path)), "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    # 1. response: reco vs regressed
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(reco / true - 1, bins=100, range=(-1, 1), histtype="step",
            density=True, lw=2, label="reco / true")
    ax.hist(resp, bins=100, range=(-1, 1), histtype="step",
            density=True, lw=2, label="regressed / true")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$m / m_\mathrm{true} - 1$")
    ax.set_ylabel("normalised")
    ax.set_title("Mass response: reco vs regressed")
    ax.legend()
    fig.savefig(os.path.join(outdir, "response.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. mass distributions
    fig, ax = plt.subplots(figsize=(8, 6))
    for arr, lab in [(true, "true Higgs"), (pred, "regressed"), (reco, "reco")]:
        ax.hist(arr, bins=80, range=(40, 250), histtype="step", density=True, lw=2, label=lab)
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("normalised")
    ax.set_title("Mass distributions")
    ax.legend()
    fig.savefig(os.path.join(outdir, "mass_dists.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. response vs true mass
    bins = np.arange(50, 300, 20)
    fig, ax = plt.subplots(figsize=(8, 6))
    cx, med, lo, hi = profile(true, resp, bins)
    ax.errorbar(cx, med, yerr=[lo, hi], fmt="o-", capsize=3, label="regressed")
    cx2, med2, lo2, hi2 = profile(true, reco / true - 1, bins)
    ax.errorbar(cx2, med2, yerr=[lo2, hi2], fmt="s--", capsize=3, label="reco")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("true Higgs mass [GeV]")
    ax.set_ylabel("median response")
    ax.set_title("Response vs true mass")
    ax.legend()
    fig.savefig(os.path.join(outdir, "response_vs_mass.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nplots -> {outdir}")


if __name__ == "__main__":
    main()
