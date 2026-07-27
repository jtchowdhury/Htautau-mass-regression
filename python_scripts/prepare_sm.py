"""
prepare_sm.py
-------------
Build salt-format h5 inputs for the SM samples so the trained model can be
evaluated on them and the reco-vs-regressed mass peaks can be plotted.

Selection: R10TruthLabel_R22v1 == 16 (hadronic tautau) ONLY. Unlike the
flat-mass training prep, we do NOT cut on GhostHBosonsMass>0 — that field is
not reliably filled in the SM samples. The true Higgs mass for these samples is
the known SM value (125 GeV); supply it at plotting time with
`analyze_predictions.py --mh 125`.

Outputs (same jets/flow/tracks schema as the training h5, so salt can read them
with the same config + norm_dict):
    data/htautau_sm603700.h5   (SM HH -> bb tautau)
    data/htautau_sm603419.h5   (SM ttbar)
"""
import os
import glob
import gc
import numpy as np
import h5py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLES = {
    "sm603700": ("/data/mfujimot/tddOutput/forRegression/"
                 "user.mfujimot.603700.e8564_s4159_r15530_p6646."
                 "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5"),
    "sm603419": ("/data/mfujimot/tddOutput/forRegression/"
                 "user.mfujimot.603419.e8559_s4159_r15224_p6646."
                 "tdd.FatJets.25_2_48.26-06-08_prod_080626_output.h5"),
}

JET_VARS  = ["pt", "eta", "mass"]
EXTRA_JET = ["R10TruthLabel_R22v1"]
TARGET    = "GhostHBosonsMass"
FLOW_VARS = ["flow_pt", "flow_energy", "flow_deta", "flow_dphi", "flow_dr"]
TRACK_VARS = [
    "d0", "z0SinTheta",
    "lifetimeSignedD0", "lifetimeSignedZ0SinTheta",
    "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    "qOverP", "qOverPUncertainty",
    "numberOfPixelHits", "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
    "leptonID",
]
LABEL          = "R10TruthLabel_R22v1"
HTAUTAU_HAD    = 16
N_CONST        = 100
DEFAULT_TARGET = 125000.0   # MeV; placeholder truth, NOT used by the peak plots


def build_dtypes(sample_file):
    with h5py.File(sample_file, "r") as f:
        jn = set(f["jets"].dtype.names)
        jet_dtype = np.dtype(
            [(v, f["jets"].dtype[v] if v in jn else np.float32)
             for v in JET_VARS + [TARGET] + EXTRA_JET])
        flow_dtype  = np.dtype([(v, f["flow"].dtype[v])   for v in FLOW_VARS + ["valid"]])
        track_dtype = np.dtype([(v, f["tracks"].dtype[v]) for v in TRACK_VARS + ["valid"]])
    return jet_dtype, flow_dtype, track_dtype


def create_output(path, jet_dtype, flow_dtype, track_dtype):
    with h5py.File(path, "w") as f:
        f.create_dataset("jets",   shape=(0,),         maxshape=(None,),
                         dtype=jet_dtype,   chunks=(4096,))
        f.create_dataset("flow",   shape=(0, N_CONST), maxshape=(None, N_CONST),
                         dtype=flow_dtype,  chunks=(512, N_CONST))
        f.create_dataset("tracks", shape=(0, N_CONST), maxshape=(None, N_CONST),
                         dtype=track_dtype, chunks=(512, N_CONST))


def append(path, group, arr):
    if arr.shape[0] == 0:
        return
    with h5py.File(path, "a") as f:
        d = f[group]
        n0 = d.shape[0]
        d.resize(n0 + arr.shape[0], axis=0)
        d[n0:n0 + arr.shape[0]] = arr


def process(name, src_dir):
    files = sorted(glob.glob(src_dir + "/*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 in {src_dir}")
    out = os.path.join(DATA_DIR, f"htautau_{name}.h5")
    jet_dtype, flow_dtype, track_dtype = build_dtypes(files[0])
    create_output(out, jet_dtype, flow_dtype, track_dtype)

    total = 0
    print(f"\n=== {name}: {len(files)} file(s) ===")
    for fp in files:
        with h5py.File(fp, "r") as f:
            jn = set(f["jets"].dtype.names)
            label = f["jets"][LABEL][:]
            sel = (label == HTAUTAU_HAD)
            idx = np.where(sel)[0]
            if idx.size == 0:
                continue

            jets_sel = np.empty(idx.size, dtype=jet_dtype)
            for v in JET_VARS + [TARGET] + EXTRA_JET:
                if v in jn:
                    jets_sel[v] = f["jets"][v][:][idx]
                else:                                   # missing (e.g. GhostHBosonsMass in SM)
                    jets_sel[v] = DEFAULT_TARGET if v == TARGET else 0

            flow_sel = np.empty((idx.size, N_CONST), dtype=flow_dtype)
            for v in FLOW_VARS + ["valid"]:
                flow_sel[v] = f["flow"][v][:][idx]

            tracks_sel = np.empty((idx.size, N_CONST), dtype=track_dtype)
            for v in TRACK_VARS + ["valid"]:
                tracks_sel[v] = f["tracks"][v][:][idx]

        append(out, "jets",   jets_sel)
        append(out, "flow",   flow_sel)
        append(out, "tracks", tracks_sel)
        total += idx.size
        del jets_sel, flow_sel, tracks_sel
        gc.collect()

    with h5py.File(out, "r") as f:
        n = f["jets"].shape[0]
        assert f["flow"].shape[0] == n == f["tracks"].shape[0], "group length mismatch!"
        assert np.any(f["jets"]["pt"][:] != 0), "jet pt all zero — write failed!"
    print(f"  wrote {total:,} hadronic-tautau jets -> {out}")


def main():
    for name, src in SAMPLES.items():
        process(name, src)
    print("\n=== SM prep done ===")


if __name__ == "__main__":
    main()
