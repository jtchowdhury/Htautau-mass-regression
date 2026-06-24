# H→ττ Mass Regression

Mass regression for boosted H→ττ fat jets using the ATLAS detector simulation.
The goal is to train a transformer-based neural network (Salt framework) to correct
the reconstructed jet mass back toward the true Higgs mass.

---

## Physics Background

A boosted H→ττ jet has three distinct mass scales:

```
Truth Higgs mass (GhostHBosonsMass)       ~125 GeV     ← regression target
       ↓  neutrino loss (~30–40%)
Truth jet mass (R10TruthLabel_R22v1_TruthJetMass)     ← visible particles only
       ↓  detector smearing (~20%)
Reco jet mass (mass)                                   ← what the detector gives us
```

Mass regression tries to close both gaps. The residual
`(m_reco / m_Higgs) - 1 = 0` is a perfect correction.

---

## Samples

All samples are large-R (R=1.0) fat jets processed by Minori Fujimoto's TDD framework.
Located at `/data/mfujimot/tddOutput/forRegression/` on the UChicago AF cluster.

| DSID   | Description                  | Truth Higgs mass      | Notes                                      |
|--------|------------------------------|-----------------------|--------------------------------------------|
| 802168 | Flat-mass BSM H→ττ           | Uniform ~100–800 GeV  | Primary training sample; prevents network from always predicting 125 GeV |
| 603700 | SM HH→bbττ                   | Fixed ~125 GeV        | Standard model reference                  |
| 603419 | SM top (ttH-like)            | Assumed 125 GeV       | GhostHBosonsMass not stored; 125 GeV assumed |

Jets are selected with `R10TruthLabel_R22v1 == 16` (HtautauHad: both taus decay hadronically).

---

## HDF5 File Structure

Each file contains six top-level datasets:

| Dataset              | Shape              | Contents                                                    |
|----------------------|--------------------|-------------------------------------------------------------|
| `jets`               | (N,)               | Jet kinematics, substructure, truth labels, tagger scores, ghost association, event info |
| `tracks`             | (N, 100)           | Up to 100 charged tracks per jet with hit counts, impact parameters, truth labels |
| `flow`               | (N, 100)           | Particle flow objects per jet (pt, energy, deta, dphi, dr)  |
| `GhostHadronsFinalLabel` | (N, 5)         | Ghost-matched hadrons with flavour, pdgId, kinematics       |
| `truth_hadrons`      | (N, 5)             | Truth-level hadrons                                         |
| `cutBookkeeper`      | —                  | Event counts and sum-of-weights for normalisation           |

Key jet-level fields used in this project:

- **Kinematics**: `pt`, `eta`, `mass` (reco)
- **Truth labels**: `R10TruthLabel_R22v1`, `R10TruthLabel_R22v1_TruthJetMass`, `R10TruthLabel_R22v1_TruthJetPt`
- **Ghost association**: `GhostHBosonsMass`, `GhostHBosonsPt`, `GhostHBosonsCount`
- **Substructure**: `Tau21`, `Tau32`, `D2`, `C2`, `N2`, `KtDR`, `Split12/23`, `ECF1/2/3`, etc.
- **Tagger scores**: `GN2Xv02_*`, `GN2XTauV00_*`, `GN3XV00_*`

---

## Directory Structure

```
htautau_regression/
├── python_scripts/
│   ├── explore_htautau.py      # Exploratory plots (mass, pT, resolution, eta)
│   └── prepare_htautau.py      # Data prep: filter, split, write Salt-ready h5 files
├── job_scripts/
│   ├── run_explore.sh/.sub     # Condor job for exploratory plots
│   ├── run_prep.sh/.sub        # Condor job for data preparation
│   └── run_train.sh/.sub       # Condor job for Salt training (requires GPU)
├── htautau_regression.yaml     # Salt model config for mass regression
├── data/                       # Created by prepare_htautau.py
│   ├── htautau_train.h5
│   ├── htautau_val.h5
│   ├── htautau_test.h5
│   ├── norm_dict.yaml
│   └── class_dict.yaml
├── plots/                      # Created by explore_htautau.py
│   ├── htautau_mass_by_sample.png
│   ├── htautau_pt_by_sample.png
│   ├── htautau_resolution.png
│   ├── htautau_tjet_resolution.png
│   ├── htautau_resolution_vs_pt.png
│   ├── htautau_eta.png
│   ├── htautau_resolution_vs_eta.png
│   └── htautau_resolution_vs_higgsm.png
└── logs/                       # Condor job logs (stdout, stderr, log)
```

---

## Workflow

### Step 1 — Exploratory Plots

Produces mass distributions, pT distributions, and resolution plots for all three samples.

```bash
condor_submit job_scripts/run_explore.sub
tail -f logs/explore.out
```

### Step 2 — Prepare Training Data

Reads the 802168 flat-mass sample, selects label-16 jets, splits into
train/val/test (70/15/15), and writes Salt-compatible h5 files plus `norm_dict.yaml`.

```bash
condor_submit job_scripts/run_prep.sub
tail -f logs/prep.out
```

At the end of the log, the script prints the `norm_params` block to paste into
`htautau_regression.yaml` under the `RegressionTask`.

### Step 3 — Train

Fill in the `norm_params` values printed by Step 2, then submit the training job.

```bash
# Edit htautau_regression.yaml: replace FILL_AFTER_PREP with printed values
condor_submit job_scripts/run_train.sub
tail -f logs/train.out
```

Salt saves model checkpoints and metrics to `logs/htautau_mass_regression/`.

---

## Model Architecture

Defined in `htautau_regression.yaml`. Uses the Salt framework
(`~/salt/`) built on PyTorch Lightning.

- **Inputs**: jet substructure variables (global) + particle flow constituents (per-particle)
- **Encoder**: Transformer (6 layers, embed_dim=256, 8 heads)
- **Pooling**: Global Attention Pooling
- **Task**: `RegressionTask` predicting `GhostHBosonsMass` (truth Higgs mass in MeV)
- **Loss**: MSE

---

## Environment

```bash
conda activate myenv   # Python env with h5py, numpy, matplotlib, salt, puma-hep
```

Salt is installed at `~/salt/`. The `salt` CLI command is available after activating myenv.

---

## Condor Tips

```bash
condor_q                          # check job status (I=idle, R=running, H=held)
condor_q -better-analyze <job_id> # why is a job idle?
condor_q -held                    # see held jobs and reason
condor_rm <job_id>                # cancel a job
watch -n 10 condor_q              # live status update
```
