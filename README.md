# Htautau-mass-regression

Regression of the truth Higgs-boson mass for boosted **H → ττ (hadronic)** large-R
jets, using the ATLAS [`salt`](https://github.com/umami-hep/salt) training framework
(transformer encoder over jet constituents). The model takes jet kinematics plus
particle-flow and track constituents and predicts the per-jet Higgs mass
(`GhostHBosonsMass`).

> Status: research / development. Runs on the UChicago Analysis Facility (AF) via
> HTCondor.

---

## What it does

Given a boosted jet from an `H → τ_had τ_had` decay, reconstruct the parent Higgs
mass. Because the τ decays carry away neutrinos, the visible (reco) jet mass is
systematically low; the regression learns to recover the true mass from the full
constituent-level information.

Training uses a **flat-mass BSM sample** (uniform Higgs-mass spectrum) so the network
sees a wide mass range rather than a single 125 GeV peak.

---

## Repository layout

```
configs/
  htautau_regression.yaml     salt config: model, data, trainer
python_scripts/
  prepare_htautau.py          build train/val/test .h5 + norm_dict from source samples
  explore_htautau.py          diagnostic plots (mass/pT/resolution distributions)
job_scripts/
  run_prep.{sh,sub}           HTCondor wrappers for prepare_htautau.py
  run_explore.{sh,sub}        HTCondor wrappers for explore_htautau.py
  run_train.{sh,sub}          HTCondor wrappers for `salt fit`
data/                         generated (not tracked): *.h5, norm_dict.yaml, class_dict.yaml
logs/                         generated (not tracked): condor + training logs
Notes.txt                     working notes: sample IDs, file structure, stats
```

---

## Data

Source files are ATLAS TDD `FatJets` HDF5 outputs (path set inside the scripts):

| Sample | DSID | Role |
|--------|------|------|
| Flat-mass BSM | 802168 | training (uniform Higgs mass) |
| SM HH → bbττ | 603700 | evaluation |
| SM top | 603419 | evaluation / background |

Each source file contains `jets`, `flow`, `tracks` groups (plus truth-hadron groups).
`prepare_htautau.py`:

1. selects `H → τ_had τ_had` jets: `R10TruthLabel_R22v1 == 16` **and** `GhostHBosonsMass > 0`
2. splits 70 / 15 / 15 into train / val / test (seed 42)
3. writes `data/htautau_{train,val,test}.h5`
4. computes input normalisation over the **train** split → `data/norm_dict.yaml`
5. writes an empty `data/class_dict.yaml` (required by salt for a pure-regression task)
6. verifies every output group is non-zero and aborts loudly otherwise

---

## Model (see `configs/htautau_regression.yaml`)

- **Inputs** — jets: `pt, eta, mass`; flow: `flow_pt, flow_energy, flow_deta, flow_dphi, flow_dr`;
  tracks: impact parameters, lifetime-signed significances, `qOverP`, hit counts, `leptonID`.
- **Init nets** — per-constituent embedding to 256 dims (flow and tracks share the embedding space).
- **Encoder** — `salt.models.Transformer`, 6 layers, 8 heads, embed 256, out 128, gated dense, 8 registers.
- **Pooling** — `GlobalAttentionPooling`.
- **Task** — `RegressionTask` (`mass_regression`) predicting `GhostHBosonsMass`, MSE loss,
  dense head `[128, 64, 32]`, target standardised via `norm_params` (mean/std).
- **Optimiser** — one-cycle LR (`initial 1e-7 → max 5e-4 → end 1e-5`, `pct_start 0.01`), weight decay 1e-5.
- **Trainer** — 50 epochs, single GPU, `bf16-mixed` precision, `gradient_clip_val: 1.0`.

---

## Setup

```bash
source ~/miniforge3/bin/activate myenv     # conda env with salt + deps
# salt is expected at ~/salt
```

## Running (HTCondor)

Submit from the repo root. **Do not run prepare/train on the login node.**

```bash
# 1. Build the datasets  (must finish before training)
condor_submit job_scripts/run_prep.sub
condor_q <user>
cat logs/prep.out          # must end with: "OK — all groups populated."

# 2. (optional) diagnostic plots
condor_submit job_scripts/run_explore.sub

# 3. Train
condor_submit job_scripts/run_train.sub
cat logs/train.out
```

After a successful prep run, `prep.out` prints a `norm_params:` block (target mean/std).
If those differ from the values in `htautau_regression.yaml`, update them there before training.

---