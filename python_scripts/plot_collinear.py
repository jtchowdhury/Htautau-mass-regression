"""
plot_collinear.py
-----------------
Five-curve mass comparison for the flat-mass BSM sample (802168):

  1. truth Higgs mass          GhostHBosonsMass
  2. truth jet mass            R10TruthLabel_R22v1_TruthJetMass
  3. reco jet mass             mass
  4. collinear (truth jet)     m_tjet * (1 + MET / pt_tjet)
  5. collinear (reco jet)      m_reco * (1 + MET / pt_reco)

The collinear correction adds the neutrino momentum (inferred from MET) back to
the jet. Two forced assumptions (documented on the plot):
  * jet phi is not stored -> MET assumed collinear with the jet (cos d-phi = 1),
    so MET_along = MET and the |d-phi|<=pi/2 cut cannot be applied.
  * only TRUTH MET (truth_met_met) exists -> idealized estimate.

Convention: the physical collinear scaling m = m_jet*(1 + MET/pt) is used
(not the literal m_jet/x, which gives unphysically large masses -- confirm with
advisor).

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


def main():
    reco_m, reco_pt, tj_m, tj_pt, mh, met = [], [], [], [], [], []
    for fp in sorted(glob.glob(SRC + "/*.h5")):
        with h5py.File(fp, "r") as h:
            j = h["jets"]
            sel = (j["R10TruthLabel_R22v1"][:] == 16) & (j["GhostHBosonsMass"][:] > 0)
            reco_m.append(j["mass"][:][sel])
            reco_pt.append(j["pt"][:][sel])
            tj_m.append(j["R10TruthLabel_R22v1_TruthJetMass"][:][sel])
            tj_pt.append(j["R10TruthLabel_R22v1_TruthJetPt"][:][sel])
            mh.append(j["GhostHBosonsMass"][:][sel])
            met.append(j["truth_met_met"][:][sel])
    reco_m  = np.concatenate(reco_m)  / 1e3    # GeV
    reco_pt = np.concatenate(reco_pt) / 1e3
    tj_m    = np.concatenate(tj_m)    / 1e3
    tj_pt   = np.concatenate(tj_pt)   / 1e3
    mh      = np.concatenate(mh)      / 1e3
    met     = np.concatenate(met)     / 1e3

    # need positive pt for the collinear correction
    ok = (reco_pt > 0) & (tj_pt > 0) & np.isfinite(met)
    reco_m, reco_pt, tj_m, tj_pt, mh, met = (a[ok] for a in (reco_m, reco_pt, tj_m, tj_pt, mh, met))

    coll_truth = tj_m   * (1.0 + met / tj_pt)     # truth jet mass + truth jet pt
    coll_reco  = reco_m * (1.0 + met / reco_pt)   # reco jet mass + reco jet pt

    print(f"N = {mh.size:,}")
    for name, arr in [("truth Higgs", mh), ("truth jet mass", tj_m),
                      ("reco jet mass", reco_m),
                      ("collinear (truth)", coll_truth),
                      ("collinear (reco)", coll_reco)]:
        print(f"  {name:20s} median {np.median(arr):6.1f} GeV")

    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "plots", "analysis_plots")
    os.makedirs(outdir, exist_ok=True)

    curves = [
        (mh,         "truth Higgs",          "#1D9E75", "-",  2.4),
        (tj_m,       "truth jet mass",       "#2C7FB8", "--", 2.0),
        (reco_m,     "reco jet mass",        "#5B4FCF", ":",  2.0),
        (coll_truth, "collinear (truth jet)","#D85A30", "-",  2.0),
        (coll_reco,  "collinear (reco jet)", "#B00020", "-",  2.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    for arr, lab, col, ls, lw in curves:
        ax.hist(arr, bins=90, range=(40, 300), histtype="step", density=True,
                color=col, ls=ls, lw=lw, label=lab)
    ax.axvline(125, color="black", ls="--", lw=1, alpha=0.4, label="125 GeV")
    ax.set_xlabel("mass [GeV]")
    ax.set_ylabel("Normalised")
    ax.set_title("Collinear mass comparison (flat-mass BSM)\n"
                 "collinear = $m_\\mathrm{jet}(1+\\mathrm{MET}/p_T)$; truth MET, "
                 "MET $\\parallel$ jet assumed", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(outdir, "collinear_mass.png")
    fig.savefig(out, dpi=150)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()
