"""
prepare_htautau.py  (rewritten — robust single-pass version)
------------------------------------------------------------
Reads the 802168 flat-mass BSM sample, selects H->tautau hadronic jets
(R10TruthLabel_R22v1 == 16 AND GhostHBosonsMass > 0), splits into
train/val/test, and writes:

    data/htautau_train.h5
    data/htautau_val.h5
    data/htautau_test.h5
    data/norm_dict.yaml
    data/class_dict.yaml

Why this rewrite exists
-----------------------
The previous version pre-allocated zero-filled compound datasets and then
relied on writes that silently failed (or crashed mid-write), leaving the
output files entirely zero across every field and group. h5py will SILENTLY
DO NOTHING if you assign to a single field of an on-disk compound dataset
(`dset["field"][a:b] = x`). The ONLY reliable way to populate a compound
dataset is to assign a complete numpy structured array to a row slice
(`dset[a:b] = structured_array`). This version does exactly that, appending
to resizable datasets one source file at a time, and then VERIFIES the
result — the script aborts loudly if any group is still zero.

Paths can be overridden with environment variables (used by the test
harness); they default to the real cluster paths.

    HTAUTAU_SAMPLE_DIR   directory of source .h5 files
    HTAUTAU_DATA_DIR     output directory

Submit via condor (run_prep.sub) — do NOT run on the login node.
"""

import gc
import os
import glob
import numpy as np
import h5py
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_DIR = os.environ.get(
    "HTAUTAU_SAMPLE_DIR",
    "/data/mfujimot/tddOutput/forRegression/"
    "user.mfujimot.802168.e8558_s4159_r15530_p6646."
    "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5",
)
DATA_DIR = os.environ.get("HTAUTAU_DATA_DIR", os.path.join(BASE_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JET_VARS  = ["pt", "eta", "mass"]
EXTRA_JET = ["R10TruthLabel_R22v1"]          # kept for bookkeeping, not a model input
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

TARGET      = "GhostHBosonsMass"
LABEL       = "R10TruthLabel_R22v1"
HTAUTAU_HAD = 16
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
SEED        = 42
N_CONST     = 100
SPLITS      = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Output file creation (resizable, so we can append per source file)
# ---------------------------------------------------------------------------
def build_dtypes(sample_file):
    with h5py.File(sample_file, "r") as f:
        jet_dtype   = np.dtype([(v, f["jets"].dtype[v])   for v in JET_VARS + [TARGET] + EXTRA_JET])
        flow_dtype  = np.dtype([(v, f["flow"].dtype[v])   for v in FLOW_VARS + ["valid"]])
        track_dtype = np.dtype([(v, f["tracks"].dtype[v]) for v in TRACK_VARS + ["valid"]])
    return jet_dtype, flow_dtype, track_dtype


def create_outputs(jet_dtype, flow_dtype, track_dtype):
    paths = {}
    for s in SPLITS:
        p = os.path.join(DATA_DIR, f"htautau_{s}.h5")
        with h5py.File(p, "w") as f:
            f.create_dataset("jets",   shape=(0,),         maxshape=(None,),
                             dtype=jet_dtype,   chunks=(4096,))
            f.create_dataset("flow",   shape=(0, N_CONST), maxshape=(None, N_CONST),
                             dtype=flow_dtype,  chunks=(512, N_CONST))
            f.create_dataset("tracks", shape=(0, N_CONST), maxshape=(None, N_CONST),
                             dtype=track_dtype, chunks=(512, N_CONST))
        paths[s] = p
    return paths


def append(path, group, arr):
    """Append a complete structured array to a resizable dataset (one-shot,
    whole-struct write — the only reliable way to populate a compound dataset)."""
    if arr.shape[0] == 0:
        return
    with h5py.File(path, "a") as f:
        d  = f[group]
        n0 = d.shape[0]
        d.resize(n0 + arr.shape[0], axis=0)
        d[n0:n0 + arr.shape[0]] = arr


# ---------------------------------------------------------------------------
# Streaming normalisation accumulator (train split only)
# ---------------------------------------------------------------------------
class NormAccum:
    def __init__(self):
        self.acc = {
            "jets":   {v: [0.0, 0.0, 0] for v in JET_VARS + [TARGET]},
            "flow":   {v: [0.0, 0.0, 0] for v in FLOW_VARS},
            "tracks": {v: [0.0, 0.0, 0] for v in TRACK_VARS},
        }

    def _add(self, group, var, vals):
        vals = vals[np.isfinite(vals)]
        a = self.acc[group][var]
        a[0] += float(vals.sum())
        a[1] += float((vals.astype(np.float64) ** 2).sum())
        a[2] += int(vals.size)

    def add_jets(self, jets_struct):
        for v in JET_VARS + [TARGET]:
            self._add("jets", v, jets_struct[v].astype(np.float64))

    def add_const(self, group, struct, var_list):
        valid = struct["valid"]
        for v in var_list:
            self._add(group, v, struct[v][valid].astype(np.float64))

    def finalize(self):
        out = {}
        for group, vars_acc in self.acc.items():
            out[group] = {}
            for v, (s, s2, n) in vars_acc.items():
                if n == 0:
                    out[group][v] = {"mean": 0.0, "std": 1.0}
                    continue
                mean = s / n
                std  = max(float(np.sqrt(max(s2 / n - mean ** 2, 0.0))), 1e-8)
                out[group][v] = {"mean": float(mean), "std": std}
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng   = np.random.default_rng(SEED)
    files = sorted(glob.glob(SAMPLE_DIR + "/*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in: {SAMPLE_DIR}")
    print(f"Found {len(files)} source file(s) in {SAMPLE_DIR}\n")

    jet_dtype, flow_dtype, track_dtype = build_dtypes(files[0])
    out_paths = create_outputs(jet_dtype, flow_dtype, track_dtype)
    norm      = NormAccum()
    totals    = {s: 0 for s in SPLITS}

    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")
        with h5py.File(fp, "r") as f:
            label  = f["jets"][LABEL][:]
            target = f["jets"][TARGET][:]
            sel    = (label == HTAUTAU_HAD) & (target > 0) & np.isfinite(target)
            sel_idx = np.where(sel)[0]
            n_sel   = sel_idx.size
            print(f"  selected {n_sel:,} H->tautau jets")
            if n_sel == 0:
                continue

            # Per-jet split assignment
            r = rng.random(n_sel)
            split_of = np.where(r < TRAIN_FRAC, "train",
                       np.where(r < TRAIN_FRAC + VAL_FRAC, "val", "test"))

            # --- Build full structured arrays for the selected jets ---
            jets_sel = np.empty(n_sel, dtype=jet_dtype)
            for v in JET_VARS + [TARGET] + EXTRA_JET:
                jets_sel[v] = f["jets"][v][:][sel_idx]

            flow_sel = np.empty((n_sel, N_CONST), dtype=flow_dtype)
            for v in FLOW_VARS + ["valid"]:
                flow_sel[v] = f["flow"][v][:][sel_idx]

            tracks_sel = np.empty((n_sel, N_CONST), dtype=track_dtype)
            for v in TRACK_VARS + ["valid"]:
                tracks_sel[v] = f["tracks"][v][:][sel_idx]

        # --- Append per split (whole-struct writes) ---
        for s in SPLITS:
            m = (split_of == s)
            if not m.any():
                continue
            append(out_paths[s], "jets",   jets_sel[m])
            append(out_paths[s], "flow",   flow_sel[m])
            append(out_paths[s], "tracks", tracks_sel[m])
            totals[s] += int(m.sum())

        # --- Accumulate normalisation over TRAIN jets only ---
        tr = (split_of == "train")
        if tr.any():
            norm.add_jets(jets_sel[tr])
            norm.add_const("flow",   flow_sel[tr],   FLOW_VARS)
            norm.add_const("tracks", tracks_sel[tr], TRACK_VARS)

        del jets_sel, flow_sel, tracks_sel
        gc.collect()

    print(f"\nSplit totals: { {s: f'{n:,}' for s, n in totals.items()} }")

    # --- Write norm_dict + class_dict ---
    norm_dict = norm.finalize()
    norm_path = os.path.join(DATA_DIR, "norm_dict.yaml")
    with open(norm_path, "w") as f:
        yaml.dump(norm_dict, f, default_flow_style=False)
    print(f"Saved: {norm_path}")

    class_path = os.path.join(DATA_DIR, "class_dict.yaml")
    with open(class_path, "w") as f:
        yaml.dump({}, f)
    print(f"Saved: {class_path}")

    # --- VERIFY (fail loud — never ship a zero file again) ---
    print("\n=== Verifying output ===")
    for s in SPLITS:
        if totals[s] == 0:
            continue
        with h5py.File(out_paths[s], "r") as f:
            gm  = f["jets"][TARGET][:]
            pt  = f["jets"]["pt"][:]
            flw = f["flow"]["flow_pt"][:]
            trk = f["tracks"]["d0"][:]
            print(f"  {s}: jets={f['jets'].shape[0]:,}  "
                  f"target[min/max/mean]={gm.min():.0f}/{gm.max():.0f}/{gm.mean():.0f}")
            assert gm.max()  > 0, f"{s}: GhostHBosonsMass all zero — write failed!"
            assert pt.max()  > 0, f"{s}: jet pt all zero — write failed!"
            assert np.any(flw != 0), f"{s}: flow all zero — write failed!"
            assert np.any(trk != 0), f"{s}: tracks all zero — write failed!"
    print("  OK — all groups populated.")

    gm = norm_dict["jets"][TARGET]
    print("\n=== Paste into htautau_regression.yaml under RegressionTask ===")
    print("                norm_params:")
    print(f"                  mean: [{gm['mean']:.2f}]")
    print(f"                  std:  [{gm['std']:.2f}]")
    print("=================================================================")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
