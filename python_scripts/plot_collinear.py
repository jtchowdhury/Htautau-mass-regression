"""
plot_collinear.py
-----------------
Truth-level collinear Higgs-mass estimate from the fields that ARE available.

DATA LIMITATIONS (documented on the plot):
  * Jet phi is NOT stored, so Delta-phi(jet, MET) cannot be computed. We
    therefore ASSUME the MET is collinear with the jet (cos d-phi = 1) -- which
    is exactly the collinear approximation's own premise. Consequence: the
    |d-phi| <= pi/2 cut cannot be applied; all selected jets are kept.
  * Only TRUTH MET (truth_met_met) is available -> this is an idealized estimate,
    not what a reco-MET analysis would give.

FORMULA AMBIGUITY (both shown; confirm with advisor):
  * literal   : x = MET / pt_jet   ->  m = m_jet / x = m_jet * pt_jet / MET
                (comes out very large -- likely NOT the intended convention)
  * physical  : m = m_jet * (1 + MET / pt_jet)
                (the standard collinear scaling; shifts reco mass up modestly)

Reads sample 802168 directly (has truth_met_met); selection R10TruthLabel==16
and GhostHBosonsMass>0, matching the training selection.

Usage:
    python plot_collinear.py
"""
import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = ("/data/mfujimot/tddOutput/forRegression/"
       "user.mfujimot.802168.e8558_s4159_r15530_p6646."
       "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5")

C_TRUTH = "#1D9E75"; C_RECO = "#5B4FCF"; C_COLL = "#D85A30"


def main():
    pt, m, mh, met = [], [], [], []
    for fp in sorted(glob.glob(SRC + "/*.h5")):
        with h5py.File(fp, "r") as h:
            j = h["jets"]
            lab = j["R10TruthLabel_R22v1"][:]
            gm  = j["GhostHBosonsMass"][:]
            sel = (lab == 16) & (gm > 0)
            pt.append(j["pt"][:][sel])
            m.append(j["mass"][:][sel])
            mh.append(gm[sel])
            met.append(j["truth_met_met"][:][sel])
    pt  = np.concatenate(pt)  / 1e3     # GeV
    m   = np.concatenate(m)   / 1e3
    mh  = np.concatenate(mh)  / 1e3
    met = np.concatenate(met) / 1e3

    good = (pt > 0) & np.isfinite(m) & np.isfinite(met)
    pt, m, mh, met = pt[good], m[good], mh[good], met[good]

    x          = met / pt                 # cos d-phi = 1 assumed
    m_literal  = np.where(x > 0, m / x, np.nan)      # advisor's formula, literal
    m_physical = m * (1.0 + x)                        # standard collinear scaling

    print(f"N = {m.size:,}")
    print(f"  truth Higgs        median {np.median(mh):.1f} GeV")
    print(f"  reco jet mass      median {np.median(m):.1f} GeV")
    print(f"  collinear PHYSICAL median {np.median(m_physical):.1f} GeV   [m_jet*(1+MET/pt)]")
    print(f"  collinear LITERAL  median {np.nanmedian(m_literal):.1f} GeV   [m_jet*pt/MET] (check convention!)")

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "plots", "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(mh,         bins=80, range=(40, 300), histtype="step", density=True,
            color=C_TRUTH, lw=2, ls="-",  label="truth Higgs")
    ax.hist(m,          bins=80, range=(40, 300), histtype="step", density=True,
            color=C_RECO,  lw=2, ls=":",  label="reco jet mass")
    ax.hist(m_physical, bins=80, range=(40, 300), histtype="step", density=True,
            color=C_COLL,  lw=2, ls="-",
            label=r"collinear $m_\mathrm{jet}(1+\mathrm{MET}/p_T)$")
    ax.axvline(125, color="black", ls="--", lw=1, alpha=0.5, label="125 GeV")
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title("Truth-level collinear mass (flat-mass BSM)\n"
                 "assumes MET $\\parallel$ jet (no jet $\\phi$); truth MET",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "collinear_mass.png")
    fig.savefig(out, dpi=150)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
