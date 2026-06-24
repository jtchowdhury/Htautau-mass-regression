"""
explore_htautau.py
------------------
Exploratory plots for H->tautau fat jets:
  1. Mass distributions (truth Higgs / reco / truth jet) — per sample
  2. pT distributions   (truth Higgs / reco / truth jet) — per sample
  3. Resolution: (m_reco / m_truth_Higgs) - 1 — both samples overlaid
  4. Resolution vs reco jet pT (profile plot) — both samples overlaid

Usage:
    conda activate myenv
    cd ~/htautau_regression
    python explore_htautau.py

Output:
    htautau_mass_by_sample.png
    htautau_pt_by_sample.png
    htautau_resolution.png
    htautau_resolution_vs_pt.png
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# Base directory = parent of python_scripts/
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLES = {
    "Flat-mass BSM (802168)": (
        "/data/mfujimot/tddOutput/forRegression/"
        "user.mfujimot.802168.e8558_s4159_r15530_p6646."
        "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5"
    ),
    "SM HH→bbττ (603700)": (
        "/data/mfujimot/tddOutput/forRegression/"
        "user.mfujimot.603700.e8564_s4159_r15530_p6646."
        "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5"
    ),
    "SM top (603419)": (
        "/data/mfujimot/tddOutput/forRegression/"
        "user.mfujimot.603419.e8559_s4159_r15224_p6646."
        "tdd.FatJets.25_2_48.26-06-08_prod_080626_output.h5"
    ),
}

FIELDS = [
    "mass",
    "pt",
    "eta",
    "abs_eta",
    "R10TruthLabel_R22v1",
    "R10TruthLabel_R22v1_TruthJetMass",
    "R10TruthLabel_R22v1_TruthJetPt",
    "GhostHBosonsMass",
    "GhostHBosonsPt",
]

# Color per sample
COLORS = {
    "Flat-mass BSM (802168)": "#1D9E75",  # teal
    "SM HH→bbττ (603700)":    "#D85A30",  # coral
    "SM top (603419)":         "#5B4FCF",  # purple
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

# When a field is missing from a sample, fill with this value instead of 0.
# GhostHBosonsMass = 125000 MeV means "assume SM Higgs mass" for samples
# (like the SM top sample 603419) that don't store it.
FIELD_DEFAULTS = {
    "GhostHBosonsMass": 125000.0,  # MeV — assume 125 GeV for SM Higgs samples
}

def load_dir(path, fields):
    files = sorted(glob.glob(path + "/*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files in: {path}")
    print(f"  {len(files)} file(s)")
    chunks = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            ds = f["jets"]
            available = set(ds.dtype.names)
            dtype = []
            for fn in fields:
                if fn in available:
                    dtype.append((fn, ds.dtype[fn]))
                else:
                    default = FIELD_DEFAULTS.get(fn, 0.0)
                    print(f"  WARNING: field '{fn}' missing in {os.path.basename(fp)}, "
                          f"filling with {default}")
                    dtype.append((fn, np.float32))
            arr = np.empty(ds.shape[0], dtype=dtype)
            for fn, _ in dtype:
                if fn in available:
                    arr[fn] = ds[fn][:]
                else:
                    arr[fn] = FIELD_DEFAULTS.get(fn, 0.0)
        chunks.append(arr)
    combined = np.concatenate(chunks)
    print(f"  Total: {len(combined):,} jets")
    return combined


def select_htautau(jets):
    """
    Select H->tautau hadronic jets using the R10 truth label.
    Label 16 = HtautauHad (both taus decay hadronically).
    This is consistent across all three samples.
    """
    mask = jets["R10TruthLabel_R22v1"] == 16
    d = {
        "reco_mass":  jets["mass"][mask] / 1e3,
        "tjet_mass":  jets["R10TruthLabel_R22v1_TruthJetMass"][mask] / 1e3,
        "higgs_mass": jets["GhostHBosonsMass"][mask] / 1e3,  # 125 GeV if field was missing
        "reco_pt":    jets["pt"][mask] / 1e3,
        "tjet_pt":    jets["R10TruthLabel_R22v1_TruthJetPt"][mask] / 1e3,
        "higgs_pt":   jets["GhostHBosonsPt"][mask] / 1e3,
        "reco_eta":   jets["abs_eta"][mask],
    }
    valid = (
        (d["tjet_mass"] > 0) & np.isfinite(d["tjet_mass"]) &
        (d["reco_mass"] > 0) & np.isfinite(d["reco_mass"]) &
        (d["reco_pt"]   > 0) & np.isfinite(d["reco_pt"])
    )
    return {k: v[valid] for k, v in d.items()}


def profile(x, y, bins):
    """
    Compute median and IQR of y in bins of x.
    IQR = interquartile range (75th - 25th percentile) — robust measure of width.
    Returns (bin centers, medians, lower error, upper error).
    """
    centers, medians, lo, hi = [], [], [], []
    for i in range(len(bins) - 1):
        in_bin = (x >= bins[i]) & (x < bins[i+1])
        if in_bin.sum() < 20:  # skip bins with too few jets
            continue
        q25, q50, q75 = np.percentile(y[in_bin], [25, 50, 75])
        centers.append((bins[i] + bins[i+1]) / 2)
        medians.append(q50)
        lo.append(q50 - q25)
        hi.append(q75 - q50)
    return np.array(centers), np.array(medians), np.array(lo), np.array(hi)


# ---------------------------------------------------------------------------
# Load all samples
# ---------------------------------------------------------------------------
data = {}
for label, path in SAMPLES.items():
    print(f"\nLoading {label}:")
    jets = load_dir(path, FIELDS)
    data[label] = select_htautau(jets)
    d = data[label]
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    print(f"  H->tautau jets after cuts: {len(d['reco_mass']):,}")
    print(f"  Jets with valid Higgs mass: {hv.sum():,}")
    print(f"  TruthJet mean: {d['tjet_mass'].mean():.1f}  Reco mean: {d['reco_mass'].mean():.1f} GeV")
    if hv.sum() > 0:
        print(f"  Higgs mass mean: {d['higgs_mass'][hv].mean():.1f} GeV")
        print(f"  Neutrino loss: {1 - d['tjet_mass'][hv].mean()/d['higgs_mass'][hv].mean():.1%}")
    print(f"  Detector loss: {1 - d['reco_mass'].mean()/d['tjet_mass'].mean():.1%}")


# ---------------------------------------------------------------------------
# Plot 1: Mass distributions — per sample, three panels
# Suppress truth Higgs for SM samples (fixed 125 GeV spike dominates)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
SHOW_HIGGS = [True, False, False]

for ax, (label, d), show_higgs in zip(axes, data.items(), SHOW_HIGGS):
    color = COLORS[label]
    if show_higgs:
        ax.hist(d["higgs_mass"], bins=80, range=(40, 250), histtype="step",
                density=True, color=color, lw=2, ls="-",  label="truth Higgs")
    ax.hist(d["reco_mass"],  bins=80, range=(40, 250), histtype="step",
            density=True, color=color, lw=2, ls=":",  label="reco")
    ax.hist(d["tjet_mass"],  bins=80, range=(40, 250), histtype="step",
            density=True, color=color, lw=2, ls="--", label="truth jet")
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title(label)
    ax.legend(fontsize=10)

fig.suptitle(r"H$\to\tau\tau$: truth Higgs (solid) / reco (dotted) / truth jet (dashed)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_mass_by_sample.png"), dpi=150)
print("\nSaved: htautau_mass_by_sample.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 2: pT distributions — per sample, three panels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ax, (label, d) in zip(axes, data.items()):
    color = COLORS[label]
    ax.hist(d["higgs_pt"], bins=80, range=(200, 1500), histtype="step",
            density=True, color=color, lw=2, ls="-",  label="truth Higgs")
    ax.hist(d["reco_pt"],  bins=80, range=(200, 1500), histtype="step",
            density=True, color=color, lw=2, ls=":",  label="reco")
    ax.hist(d["tjet_pt"],  bins=80, range=(200, 1500), histtype="step",
            density=True, color=color, lw=2, ls="--", label="truth jet")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title(label)
    ax.legend(fontsize=10)

fig.suptitle(r"H$\to\tau\tau$ $p_T$: truth Higgs (solid) / reco (dotted) / truth jet (dashed)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_pt_by_sample.png"), dpi=150)
print("Saved: htautau_pt_by_sample.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3: Resolution (m_reco / m_truth_Higgs) - 1 — both samples overlaid
# A perfect regression would give a spike at 0. The peak position tells you the bias, the width tells you the resolution.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for label, d in data.items():
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    if hv.sum() == 0:
        continue
    response = d["reco_mass"][hv] / d["higgs_mass"][hv] - 1
    color = COLORS[label]
    median = np.median(response)
    iqr = np.percentile(response, 75) - np.percentile(response, 25)
    ax.hist(response, bins=100, range=(-1, 1), histtype="step",
            density=True, color=color, lw=2,
            label=f"{label}  (median={median:.2f}, IQR={iqr:.2f})")

ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="perfect response")
ax.set_xlabel(r"$(m_\mathrm{reco} / m_\mathrm{truth\,Higgs}) - 1$")
ax.set_ylabel("Normalised")
ax.set_title(r"Mass resolution: reco vs truth Higgs")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_resolution.png"), dpi=150)
print("Saved: htautau_resolution.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 3b: Truth jet resolution — (m_tjet / m_Higgs) - 1
# Same as Plot 3 but using truth jet mass instead of reco mass.
# Shows how much of the mass loss is from neutrinos alone (before detector effects).
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for label, d in data.items():
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    if hv.sum() == 0:
        continue
    response = d["tjet_mass"][hv] / d["higgs_mass"][hv] - 1
    color = COLORS[label]
    median = np.median(response)
    iqr = np.percentile(response, 75) - np.percentile(response, 25)
    ax.hist(response, bins=100, range=(-1, 1), histtype="step",
            density=True, color=color, lw=2,
            label=f"{label}  (median={median:.2f}, IQR={iqr:.2f})")

ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5, label="perfect response")
ax.set_xlabel(r"$(m_\mathrm{truth\,jet} / m_\mathrm{truth\,Higgs}) - 1$")
ax.set_ylabel("Normalised")
ax.set_title(r"Truth jet resolution: neutrino loss only (no detector)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_tjet_resolution.png"), dpi=150)
print("Saved: htautau_tjet_resolution.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 4: Resolution vs reco jet pT (profile plot) — both samples overlaid
#
# Shows whether the bias and resolution depend on pT.
# Median = bias (should be flat at 0 ideally)
# Error bars = IQR (should be narrow and flat)
# ---------------------------------------------------------------------------
PT_BINS = np.arange(200, 1300, 100)  # 100 GeV wide bins from 200 to 1200 GeV

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for label, d in data.items():
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    if hv.sum() == 0:
        continue
    response = d["reco_mass"][hv] / d["higgs_mass"][hv] - 1
    color = COLORS[label]
    centers, medians, lo, hi = profile(d["reco_pt"][hv], response, PT_BINS)
    axes[0].errorbar(centers, medians, yerr=[lo, hi],
                     fmt="o-", color=color, lw=2, ms=4, label=label, capsize=3)

    centers_t, medians_t, lo_t, hi_t = profile(d["tjet_pt"][hv], response, PT_BINS)
    axes[1].errorbar(centers_t, medians_t, yerr=[lo_t, hi_t],
                     fmt="o-", color=color, lw=2, ms=4, label=label, capsize=3)

axes[0].axhline(0, color="black", ls="--", lw=1, alpha=0.5)
axes[0].set_xlabel("Reco jet $p_T$ [GeV]")
axes[0].set_ylabel(r"Median of $(m_\mathrm{reco}/m_\mathrm{Higgs} - 1)$")
axes[0].set_title("Bias vs reco $p_T$")
axes[0].legend(fontsize=10)

axes[1].axhline(0, color="black", ls="--", lw=1, alpha=0.5)
axes[1].set_xlabel("Truth Higgs $p_T$ [GeV]")
axes[1].set_ylabel(r"Median of $(m_\mathrm{reco}/m_\mathrm{Higgs} - 1)$")
axes[1].set_title("Bias vs true jet $p_T$")
axes[1].legend(fontsize=10)

fig.suptitle(r"H$\to\tau\tau$ mass resolution vs $p_T$", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_resolution_vs_pt.png"), dpi=150)
print("Saved: htautau_resolution_vs_pt.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 5: Eta distributions — both samples overlaid on one panel
# We use |eta| (absolute value) since the detector is symmetric around eta=0
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

for label, d in data.items():
    color = COLORS[label]
    ax.hist(d["reco_eta"], bins=50, range=(0, 2.5), histtype="step",
            density=True, color=color, lw=2, label=label)

ax.set_xlabel(r"$|\eta|$")
ax.set_ylabel("Normalised")
ax.set_title(r"H$\to\tau\tau$ jet $|\eta|$ distributions")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_eta.png"), dpi=150)
print("Saved: htautau_eta.png")
plt.close()


# ---------------------------------------------------------------------------
# Plot 6: Resolution vs eta — both samples overlaid, two panels
# Resolution vs reco eta
# We use |eta| bins since response is symmetric
# ---------------------------------------------------------------------------
ETA_BINS = np.linspace(0, 2.5, 11)  # 0.25-wide bins from 0 to 2.5

fig, ax = plt.subplots(figsize=(10, 6))

for label, d in data.items():
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    if hv.sum() == 0:
        continue
    response = d["reco_mass"][hv] / d["higgs_mass"][hv] - 1
    color = COLORS[label]
    centers, medians, lo, hi = profile(d["reco_eta"][hv], response, ETA_BINS)
    ax.errorbar(centers, medians, yerr=[lo, hi],
                fmt="o-", color=color, lw=2, ms=4, label=label, capsize=3)

ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xlabel(r"Reco jet $|\eta|$")
ax.set_ylabel(r"Median of $(m_\mathrm{reco}/m_\mathrm{Higgs} - 1)$")
ax.set_title(r"Bias vs $|\eta|$")
ax.legend(fontsize=10)

fig.suptitle(r"H$\to\tau\tau$ mass resolution vs $|\eta|$", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_resolution_vs_eta.png"), dpi=150)
print("Saved: htautau_resolution_vs_eta.png")
plt.close()

# ---------------------------------------------------------------------------
# Plot 7: Resolution vs truth Higgs mass (profile plot) — both samples overlaid
#
# This is the key diagnostic Valerio asked about.
# If the bias depends on Higgs mass, that explains the difference between
# the two samples (flat-mass has wide mass range, SM is fixed at 125 GeV).
# ---------------------------------------------------------------------------
MASS_BINS = np.arange(50, 500, 25)  # 25 GeV wide bins from 50 to 500 GeV

fig, ax = plt.subplots(figsize=(10, 6))

for label, d in data.items():
    hv = (d["higgs_mass"] > 0) & np.isfinite(d["higgs_mass"])
    if hv.sum() == 0:
        continue
    response = d["reco_mass"][hv] / d["higgs_mass"][hv] - 1
    color = COLORS[label]
    centers, medians, lo, hi = profile(d["higgs_mass"][hv], response, MASS_BINS)
    ax.errorbar(centers, medians, yerr=[lo, hi],
                fmt="o-", color=color, lw=2, ms=4, label=label, capsize=3)

ax.axhline(0, color="black", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("Truth Higgs mass [GeV]")
ax.set_ylabel(r"Median of $(m_\mathrm{reco}/m_\mathrm{Higgs} - 1)$")
ax.set_title(r"Mass bias vs truth Higgs mass")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "htautau_resolution_vs_higgsm.png"), dpi=150)
print("Saved: htautau_resolution_vs_higgsm.png")
plt.close()

print("\nAll done.")
