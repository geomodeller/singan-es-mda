# SinGAN–ES-MDA for Geological CO₂ Storage History Matching:
# From a Single Geological Interpretation to History Matching  

## A SinGAN–ES-MDA Framework for CO₂ Storage in Channelized Aquifers  
**Jo et al., Geoenergy Science and Engineering (2026)**

---

## Overview

This repository provides a full implementation of the **SinGAN–ES-MDA framework** for ensemble-based history matching in Geological Carbon Storage (GCS) applications under severe data and interpretation constraints.

The method enables ensemble construction and dynamic data assimilation starting from a **single geological interpretation**, addressing a fundamental limitation in early-stage CO₂ storage projects where multiple geological realizations are unavailable.

### Framework Components

- **SinGAN (Single-Image Generative Adversarial Network)**  
  Learns multi-scale geological statistics from one 3D facies model.

- **ES-MDA (Ensemble Smoother with Multiple Data Assimilation)**  
  Assimilates dynamic observations (injector bottom-hole pressure, BHP).

- **CMG-GEM / IMEX**  
  Performs forward multiphase flow reservoir simulations.

The workflow reduces data–model misfit while preserving geological realism.

---

## Scientific Motivation

Ensemble-based history matching methods require a prior ensemble that spans subsurface uncertainty. However, many GCS projects only possess:

- A single high-confidence geological interpretation  
- Sparse well control  
- Limited geophysical resolution  

Under these conditions, constructing statistically meaningful ensembles is impractical.

This framework enables:

- Generation of geologically plausible ensembles from a single model  
- Assimilation of dynamic data within a constrained parameter space  
- Uncertainty quantification in pressure and CO₂ plume evolution  

### Scope of Current Implementation

Corrected uncertainties:

- Channel rotation (azimuth)  
- Translation in x-direction  
- Translation in y-direction  
- Translation in z-direction  

Not modeled:

- Channel sinuosity variation  
- Net-to-gross variation  
- Internal architectural variability  

---

## Workflow Summary

1. Train SinGAN on a single geological facies model  
2. Generate an ensemble of geometric control parameters  
3. Map parameters to geological realizations via SinGAN  
4. Run CMG simulations for each realization  
5. Extract BHP from SR3 output files  
6. Compute RMSE misfit  
7. Update parameters using ES-MDA in normal-score space  
8. Repeat assimilation for multiple iterations  
9. Perform post-assimilation forecast simulations  
10. Evaluate structural similarity (SSIM) and plume evolution  

---

## Repository Structure

```bash
├── main_run_singan_esmda.py    # Full SinGAN–ES-MDA workflow
├── esmda_utils.py              # Ensemble generation & ES-MDA utilities
├── cmg_launcher.py             # Parallel CMG execution
├── cmg_sr3_reader.py           # SR3 grid property extraction
├── visual.py                   # Visualization utilities
├── statiscal_analysis.py       # SSIM & diagnostics
├── template_3d/                # CMG base files
├── template_3d_ground_truth/   # CMG base files of ground truth
├── sinGAN_script/              # Subroutines for generate new realizations 
├── sinGAN_trained_model/       # Repository of the trained SinGAN Model
├── property_modelling_part/    # Ground truth models
├── requirements.txt
└── README.md
```
---

## Installation
### Clone Repository
```bash
git clone https://github.com/geomodeller/singan-es-mda.git
cd singan-es-mda
```

### Create Python Environment
```bash
conda create -n singan-esmda python=3.12.7
conda activate singan-esmda
pip install -r requirements.txt
```
## Dependencies
### Core Python Packages
* numpy
* matplotlib
* scipy
* tqdm
* scikit-learn
* seaborn
* torch

### External Dependency
* CMG-GEM (Licensed software required)

## Running the Workflow
```bash
python main_run_singan_esmda.py
```

The script will:
 * Create iteration directories
 * Run CMG simulations
 * Perform ES-MDA updates
 * Save figures and results

## Citation

If you use this repository, please cite:
```bibtex
@article{Jo2026SinGANESMDA,
  title={From a Single Geological Interpretation to History Matching:
         A SinGAN–ES-MDA Framework for CO₂ Storage in Channelized Aquifers},
  author={Jo, Honggeun and Park, Eunsil and Ahn, Seongin},
  journal={Geoenergy Science and Engineering},
  year={2026}
}
```
## Authors
* Honggeun Jo — Inha University
* Eunsil Park — Inha University
* Seongin Ahn — Korea Institute of Geoscience and Mineral Resources (KIGAM)

## Disclaimer

This repository is intended for research purposes only.
A valid CMG simulator license is required to run forward simulations.


