"""
prepare_htautau.py
------------------
Reads the 802168 flat-mass BSM sample, selects H->tautau hadronic jets
(R10TruthLabel_R22v1 == 16), splits into train/val/test, and writes:

    data/htautau_train.h5
    data/htautau_val.h5
    data/htautau_test.h5
    data/norm_dict.yaml    <- Salt uses this to normalise inputs
    data/class_dict.yaml   <- empty for pure regression, required by Salt

Memory-efficient: reads flow only for the selected jets (not all 942k per file).
Submit via condor (run_prep.sub) — do NOT run on the login node.
"""

import os
import glob
import numpy as np
import h5py
import yaml

# Base directory = parent of python_scripts/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_DIR = (
    "/data/mfujimot/tddOutput/forRegression/"
    "user.mfujimot.802168.e8558_s4159_r15530_p6646."
    "tdd.FatJets.25_2_56.26-05-16_prod_160526_output.h5"
)

JET_VARS = ["pt", "eta", "mass"]

FLOW_VARS = ["flow_pt", "flow_energy", "flow_deta", "flow_dphi", "flow_dr"]

TRACK_VARS = [
    "d0", "z0SinTheta",                              # decay vertex — unique to tracks
    "qOverP", "qOverPUncertainty",                   # charge + track fit quality
    "numberOfPixelHits", "numberOfSCTHits",          # track quality
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "leptonID",                                      # e/mu ID — not in flow
]

TARGET      = "GhostHBosonsMass"
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
SEED        = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_files():
    out = {}
    for split in ("train", "val", "test"):
        path = os.path.join(DATA_DIR, f"htautau_{split}.h5")
        # Open fresh (overwrite if exists)
        with h5py.File(path, "w") as f:
            pass  # create empty file
        out[split] = path
    return out


def append_to_h5(path, jets_chunk, flow_chunk, tracks_chunk):
    """Append a chunk of jets/flow/tracks to a resizable h5 dataset."""
    with h5py.File(path, "a") as f:
        for name, data in [("jets", jets_chunk), ("flow", flow_chunk), ("tracks", tracks_chunk)]:
            if name not in f:
                maxshape = (None,) + data.shape[1:]
                f.create_dataset(name, data=data, maxshape=maxshape,
                                 chunks=True, compression="lzf")
            else:
                ds = f[name]
                old_n = ds.shape[0]
                ds.resize(old_n + len(data), axis=0)
                ds[old_n:] = data


def make_jet_record(jets, jet_vars, target):
    """Build output structured array with only the fields Salt needs."""
    dtype = (
        [(v, np.float32) for v in jet_vars]
        + [(target, np.float32)]
        + [("R10TruthLabel_R22v1", np.int32)]
    )
    out = np.empty(len(jets), dtype=dtype)
    for v in jet_vars:
        out[v] = jets[v].astype(np.float32)
    out[target] = jets[target].astype(np.float32)
    out["R10TruthLabel_R22v1"] = jets["R10TruthLabel_R22v1"].astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)

    files = sorted(glob.glob(SAMPLE_DIR + "/*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files in {SAMPLE_DIR}")
    print(f"Found {len(files)} file(s)\n")

    out_paths = get_output_files()

    # Running accumulators for norm_dict (training set only)
    norm_accum = {
        "jets":   {v: {"sum": 0.0, "sum2": 0.0, "n": 0} for v in JET_VARS + [TARGET]},
        "flow":   {v: {"sum": 0.0, "sum2": 0.0, "n": 0} for v in FLOW_VARS},
        "tracks": {v: {"sum": 0.0, "sum2": 0.0, "n": 0} for v in TRACK_VARS},
    }

    total_written = {"train": 0, "val": 0, "test": 0}

    for fp in files:
        print(f"Processing {os.path.basename(fp)} ...")

        with h5py.File(fp, "r") as f:
            # Step 1: read jets only (manageable ~400 MB)
            jets_all = f["jets"][:]

            # Step 2: select label-16 jets with valid Higgs mass
            mask = (jets_all["R10TruthLabel_R22v1"] == 16) & (jets_all[TARGET] > 0)
            sel_idx = np.where(mask)[0]  # indices into this file
            print(f"  {len(sel_idx):,} H->tautau jets selected")

            if len(sel_idx) == 0:
                continue

            jets_sel = jets_all[sel_idx]
            del jets_all  # free memory immediately

            # Step 3: read flow and tracks ONLY for selected jets
            flow_sel   = f["flow"][sel_idx]
            tracks_sel = f["tracks"][sel_idx]

        # Step 4: assign each jet to train/val/test randomly
        r = rng.random(len(jets_sel))
        is_train = r < TRAIN_FRAC
        is_val   = (r >= TRAIN_FRAC) & (r < TRAIN_FRAC + VAL_FRAC)
        is_test  = r >= TRAIN_FRAC + VAL_FRAC

        for split, mask_split in [("train", is_train), ("val", is_val), ("test", is_test)]:
            if mask_split.sum() == 0:
                continue
            jchunk = make_jet_record(jets_sel[mask_split], JET_VARS, TARGET)
            fchunk = flow_sel[mask_split]
            tchunk = tracks_sel[mask_split]
            append_to_h5(out_paths[split], jchunk, fchunk, tchunk)
            total_written[split] += mask_split.sum()

        # Step 5: accumulate norm stats from training jets only
        jets_train   = jets_sel[is_train]
        flow_train   = flow_sel[is_train]
        tracks_train = tracks_sel[is_train]
        for v in JET_VARS + [TARGET]:
            vals = jets_train[v].astype(np.float64)
            vals = vals[np.isfinite(vals)]
            norm_accum["jets"][v]["sum"]  += vals.sum()
            norm_accum["jets"][v]["sum2"] += (vals ** 2).sum()
            norm_accum["jets"][v]["n"]    += len(vals)
        valid_flow = flow_train["valid"]
        for v in FLOW_VARS:
            vals = flow_train[v].astype(np.float64)[valid_flow]
            vals = vals[np.isfinite(vals)]
            norm_accum["flow"][v]["sum"]  += vals.sum()
            norm_accum["flow"][v]["sum2"] += (vals ** 2).sum()
            norm_accum["flow"][v]["n"]    += len(vals)
        valid_trk = tracks_train["valid"]
        for v in TRACK_VARS:
            vals = tracks_train[v].astype(np.float64)[valid_trk]
            vals = vals[np.isfinite(vals)]
            norm_accum["tracks"][v]["sum"]  += vals.sum()
            norm_accum["tracks"][v]["sum2"] += (vals ** 2).sum()
            norm_accum["tracks"][v]["n"]    += len(vals)

        print(f"  Written: train={is_train.sum():,}  val={is_val.sum():,}  test={is_test.sum():,}")

    # ---------------------------------------------------------------------------
    # Finalise norm_dict
    # ---------------------------------------------------------------------------
    print("\n=== Finalising norm_dict ===")
    norm_dict = {}
    for group in ("jets", "flow", "tracks"):
        norm_dict[group] = {}
        for v, acc in norm_accum[group].items():
            n = acc["n"]
            if n == 0:
                print(f"  WARNING: no valid values for {group}/{v}, using defaults")
                norm_dict[group][v] = {"mean": 0.0, "std": 1.0}
                continue
            mean = acc["sum"] / n
            std  = max(np.sqrt(acc["sum2"] / n - mean ** 2), 1e-8)
            norm_dict[group][v] = {"mean": float(mean), "std": float(std)}

    with open(os.path.join(DATA_DIR, "norm_dict.yaml"), "w") as f:
        yaml.dump(norm_dict, f, default_flow_style=False)
    print(f"Saved: {os.path.join(DATA_DIR, 'norm_dict.yaml')}")

    with open(os.path.join(DATA_DIR, "class_dict.yaml"), "w") as f:
        yaml.dump({}, f)
    print(f"Saved: {os.path.join(DATA_DIR, 'class_dict.yaml')}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n=== Done ===")
    for split, n in total_written.items():
        print(f"  {split}: {n:,} jets")

    gm = norm_dict["jets"][TARGET]
    print("\n=== Paste this into htautau_regression.yaml under RegressionTask ===")
    print(f"                norm_params:")
    print(f"                  mean: [{gm['mean']:.2f}]")
    print(f"                  std:  [{gm['std']:.2f}]")
    print("====================================================================")


if __name__ == "__main__":
    main()
