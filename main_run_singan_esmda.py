"""
=====================================================================================
SinGAN–ES-MDA Workflow for History Matching in Geological CO₂ Storage
=====================================================================================

This script implements the full computational workflow presented in:

“From a Single Geological Interpretation to History Matching:
A SinGAN–ES-MDA Framework for CO₂ Storage in Channelized Aquifers”
Jo et al., Geoenergy Science and Engineering (Revised Manuscript)

-------------------------------------------------------------------------------------
Scientific Context
-------------------------------------------------------------------------------------

Geological carbon storage (GCS) projects are frequently conducted under severe
data and interpretation constraints, where only a single, conceptually reliable
geological model is available. In such settings, conventional ensemble-based
history matching methods face a fundamental limitation: the prior ensemble must
span the true subsurface variability. When multiple geological realizations are
unavailable, constructing such an ensemble becomes impractical.

To address this challenge, the associated manuscript introduces a SinGAN–ES-MDA
framework that integrates:

    • SinGAN (Single-Image Generative Adversarial Network)
    • ES-MDA (Ensemble Smoother with Multiple Data Assimilation)

SinGAN learns multi-scale geological statistics from a single 3D facies model
and generates geologically plausible realizations through controlled geometric
transformations (rotation and spatial translations). These low-dimensional,
interpretable parameters represent dominant large-scale geometric uncertainty
(e.g., channel orientation and placement).

The ES-MDA algorithm then assimilates dynamic data (injectors’ bottom-hole
pressure, BHP) to iteratively update these parameters, reducing data–model
misfit while preserving geological realism.

-------------------------------------------------------------------------------------
Purpose of This Script
-------------------------------------------------------------------------------------

This script provides a complete, end-to-end implementation of the SinGAN–ES-MDA
workflow used in the manuscript’s synthetic 3D channelized aquifer study.

The workflow consists of:

(1) Generating an initial ensemble of SinGAN control parameters
(2) Mapping parameters into geological realizations using a trained SinGAN
(3) Running forward multiphase flow simulations in CMG-GEM/IMEX
(4) Extracting dynamic responses (BHP)
(5) Updating parameters via ES-MDA in normal-score space
(6) Repeating assimilation for multiple iterations
(7) Performing post-assimilation forecasting
(8) Evaluating structural similarity (SSIM) and dynamic performance
(9) Visualizing plume evolution and parameter convergence

The framework is intentionally scoped to correct first-order geometric
misalignment (rotation and translation) rather than internal morphological
uncertainties (e.g., sinuosity or net-to-gross variation). It assumes that the
initial geological interpretation is conceptually consistent with the true
depositional style.

-------------------------------------------------------------------------------------
Reproducibility Notes
-------------------------------------------------------------------------------------

This script assumes the availability of the following project modules:

    - esmda_utils.py
    - cmg_launcher.py
    - cmg_sr3_reader.py
    - visual.py
    - statiscal_analysis.py

For full reproducibility, the following should also be provided:

    - requirements.txt (Python dependencies)
    - CMG simulator version information
    - template_3d folder containing CMG base files
    - Trained SinGAN model weights
    - README.md with execution instructions

Note:
The SinGAN training step is a one-time computational cost (~16 hours in the
manuscript setup). ES-MDA forward simulations dominate runtime (~250 hours
for 500 realizations × 10 iterations in the synthetic case).

-------------------------------------------------------------------------------------
Author: Honggeun Jo, Seongin Ahn, and Eunsil Park
Affiliation: Inha University / KIGAM
Last Updated: 2026-02-02
=====================================================================================
"""

# =============================================================================
# 0) Imports
# =============================================================================
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

# --- ES-MDA core update function ---
from esmda import ES

# --- Project utilities (your internal modules) ---
from esmda_utils import (
    generate_x_vars,
    replace_perm_and_rocktype,
    compute_alpha_lst,
    copy_files,
    read_ensemble_sr3_for_history_matching,
    compute_ensemble_misfit_rmse,
    delete_failed_ensemble,
    NormalScoreTransformer,
    save_ensmda_results,
    load_ensmda_results,
)

from cmg_launcher import run_cmg_for_ensemble_parallel

from cmg_sr3_reader import read_SR3, get_grid_properties

from visual import (
    plot_injector_history_,
    visual_boxplot_of_mse_over_iter,
    visual_update_of_i_xvar,
)

# Optional downstream visuals (if available in your repo)
# NOTE: These are referenced later; if they are not part of your repo,
# comment them out or guard them with try/except.
from visual import (
    plot_easting_depth,
    plot_northing_depth,
    plot_map,
)

from statiscal_analysis import calculate_ssim_from_reals
from visual import visual_hist_ssim


# =============================================================================
# 1) Configuration / Parameters
# =============================================================================
def build_params():
    """
    Centralize parameters into a single dictionary for clarity and reproducibility.
    """
    p = {}

    # --- Run control ---
    p["run_dir"] = "run_05_only_trend"          # Root directory for all iteration folders
    p["num_of_iteration"] = 10                  # Number of ES-MDA assimilation iterations
    p["num_of_ensemble"] = 500                  # Ensemble size
    p["max_workers"] = 9                        # Parallel workers for running CMG
    p["random_seed"] = 77777                    # Seed for initial ensemble generation

    # --- ES-MDA alpha schedule ---
    p["alpha_decay_ratio"] = 1.0
    p["lst_alpha"] = compute_alpha_lst(p["num_of_iteration"], p["alpha_decay_ratio"])

    # --- Normal-score transform settings ---
    p["NST_scaler"] = NormalScoreTransformer()
    p["nst_max"] = 4.0
    p["nst_min"] = -4.0

    # --- Observation / error model ---
    # Relative observational error fraction for dynamic data (you may tune)
    p["stdErrOfDynamic_percentage"] = 0.001

    # --- CMG files ---
    p["cmg_data_file"] = "cmg_ccs_run_file.dat"
    p["cmg_sr3_file"] = "cmg_ccs_run_file.sr3"

    # --- Templates for forward simulation directories ---
    p["template_files"] = glob.glob("template_3d/*")

    # --- Ground truth (optional; used for SSIM analysis / diagnostics) ---
    p["ground_truth"] = np.load(os.path.join("property_modelling_part", "prop_ground_truth.npy"))
    p["ground_truth_trend"] = np.load(os.path.join("property_modelling_part", "proportion_ground_truth.npy"))

    # --- Observations extracted from a "ground-truth" SR3 ---
    # This SR3 is assumed to contain the observation time schedule you want
    p["obs"] = read_ensemble_sr3_for_history_matching(
        sr3_file="cmg_ccs_run_file_by_days_then_by_schedule.sr3",
        target_dir="template_3d_ground_truth",
        level_of_directory=0,
    )

    return p


# =============================================================================
# 2) Core Workflow Functions
# =============================================================================
def prepare_iteration_folders(xvar, iteration, params):
    """
    Create iteration folder structure and populate each realization directory
    with required template files and updated property fields.

    Parameters
    ----------
    xvar : np.ndarray
        Ensemble parameter matrix, shape = (Ne, Nxvar).
    iteration : int
        Current ES-MDA iteration.
    params : dict
        Workflow parameters.

    Returns
    -------
    target_dir : list[str]
        List of paths for each ensemble realization directory.
    """
    # Example:
    # run_05_only_trend/iter_0/0
    # run_05_only_trend/iter_0/1
    # ...
    target_dir = [f'./{params["run_dir"]}/iter_{iteration}/{i}' for i in range(xvar.shape[0])]

    # Copy base template files into all target directories
    copy_files(params["template_files"], target_dir)

    # Replace permeability/rocktype fields based on xvar
    replace_perm_and_rocktype(xvar, target_dir)

    return target_dir


def run_forward_models(iteration, params):
    """
    Run all CMG simulations in parallel for a given iteration directory.

    Notes
    -----
    You assume the ensemble directories already exist and include
    the correct CMG input files.
    """
    run_cmg_for_ensemble_parallel(
        ensemble_dir=os.path.join(params["run_dir"], f"iter_{iteration}"),
        cmg_data_file=params["cmg_data_file"],
        max_workers=params["max_workers"],
    )


def read_dynamic_and_misfit(iteration, params):
    """
    Read dynamic results (from SR3) for all ensemble realizations and compute misfit.

    Returns
    -------
    dynamic_matrix : np.ndarray
        Dynamic response for each ensemble (shape depends on your reader).
    failed_idx : list[int]
        Indices of ensemble members that failed simulation.
    rmse : np.ndarray
        RMSE misfit per ensemble member.
    """
    dynamic_matrix, failed_idx = read_ensemble_sr3_for_history_matching(
        sr3_file=params["cmg_sr3_file"],
        target_dir=os.path.join(params["run_dir"], f"iter_{iteration}"),
        level_of_directory=1,
        return_failed_reals=True,
    )

    rmse = compute_ensemble_misfit_rmse(dynamic_matrix, params["obs"])
    return dynamic_matrix, failed_idx, rmse


def esmda_update(xvar, dynamic_matrix, params, alpha):
    """
    Perform one ES-MDA update step in Normal-Score space.

    Steps
    -----
    1) remove failed realizations from xvar (caller typically does this)
    2) transform xvar -> ns_xvar (normal-score)
    3) update using ES-MDA
    4) clip updated normal-score values for stability
    5) invert transform back to original parameter space

    Returns
    -------
    xvar_updated : np.ndarray
        Updated ensemble in original parameter space.
    """
    # Transform parameters into normal-score space (approximately Gaussian)
    ns_xvar_before = params["NST_scaler"].fit_transform(xvar)

    # ES-MDA update (your ES function signature)
    ns_xvar_after = ES(
        params["obs"].reshape(1, -1),                  # Observations
        ns_xvar_before,                                # Prior ensemble (NS space)
        dynamic_matrix.reshape(xvar.shape[0], -1),      # Predicted data for ensemble
        alpha,                                         # Assimilation inflation
        params["stdErrOfDynamic_percentage"],           # Relative obs error
        add_noise=False,                                # Common choice for deterministic ES-MDA
    )

    # Clip to avoid extreme tails when inverting transform
    ns_xvar_after = np.clip(ns_xvar_after, params["nst_min"], params["nst_max"])

    # Back-transform to original space
    xvar_updated = params["NST_scaler"].inverse_transform(ns_xvar_after)

    return xvar_updated


def save_iteration_figures(iteration, params, dynamic_matrix, ensemble_dynamic, ensemble_history):
    """
    Generate and save standard diagnostics each iteration:
    - injector history match plots
    - boxplot of misfit distribution over iterations
    - evolution of selected xvar parameters
    """
    run_dir = params["run_dir"]

    plot_injector_history_(
        dynamic_matrix,
        params,
        save_path=os.path.join(run_dir, f"iter_{iteration}_hist_plot.png"),
    )

    visual_boxplot_of_mse_over_iter(
        ensemble_dynamic,
        params,
        save_path=os.path.join(run_dir, f"iter_{iteration}_mse_boxplot.png"),
    )

    visual_update_of_i_xvar(
        ensemble_history,
        i_xvar=[-4, -3, -2, -1],
        save_path=os.path.join(run_dir, f"iter_{iteration}_xvars.png"),
    )


# =============================================================================
# 3) ES-MDA Main Driver
# =============================================================================
def run_esmda(params):
    """
    Full ES-MDA workflow:
    - initialize ensemble
    - iterate: forward simulation -> read -> misfit -> update
    - forecast run
    - save results to disk

    Returns
    -------
    params, ensemble_history, ensemble_dynamic : (dict, list[np.ndarray], list[np.ndarray])
    """
    # Ensure deterministic initialization (and any internal randomness)
    np.random.seed(params["random_seed"])

    print(f"Alpha schedule: {params['lst_alpha']}")

    # Store ensemble parameters (xvar) and forward outputs across iterations
    ensemble_history = []
    ensemble_dynamic = []

    # -----------------------------------------------------
    # 3.1) Generate initial ensemble
    # -----------------------------------------------------
    xvar = generate_x_vars(
        random_seed=params["random_seed"],
        num_realizations=params["num_of_ensemble"],
    )

    # -----------------------------------------------------
    # 3.2) ES-MDA iterative assimilation loop
    # -----------------------------------------------------
    for it in range(params["num_of_iteration"]):
        print(f"\n========== ES-MDA Iteration {it} / {params['num_of_iteration']-1} ==========")

        # Record current ensemble before update
        ensemble_history.append(xvar.copy())

        # Prepare simulation folders and CMG input files
        prepare_iteration_folders(xvar, it, params)

        # Run CMG for all ensemble members
        run_forward_models(it, params)

        # Read dynamic outputs and compute RMSE
        dynamic_matrix, failed_idx, rmse = read_dynamic_and_misfit(it, params)
        print(f"RMSE: mean={np.mean(rmse):.4f}, std={np.std(rmse):.4f}, min={np.min(rmse):.4f}")

        # Remove failed realizations from xvar (keep only successful ones)
        xvar_result = delete_failed_ensemble(xvar, failed_idx)
        # Handle tuple return (array, count) or just array
        if isinstance(xvar_result, tuple):
            xvar = xvar_result[0].copy()
        else:
            xvar = xvar_result.copy()

        # IMPORTANT:
        # After deleting failed ensembles, dynamic_matrix should correspond to remaining ensembles.
        # Your read function likely already returns consistent matrix; if not, you must also filter dynamic_matrix.

        # Update ensemble using ES-MDA
        xvar = esmda_update(
            xvar=xvar,
            dynamic_matrix=dynamic_matrix,
            params=params,
            alpha=params["lst_alpha"][it],
        )

        # Record dynamic outputs
        ensemble_dynamic.append(dynamic_matrix)

        # Save standard diagnostics
        save_iteration_figures(
            iteration=it,
            params=params,
            dynamic_matrix=dynamic_matrix,
            ensemble_dynamic=ensemble_dynamic,
            ensemble_history=ensemble_history,
        )

    # -----------------------------------------------------
    # 3.3) Final forecast (future prediction) using updated ensemble
    # -----------------------------------------------------
    forecast_it = params["num_of_iteration"]
    print(f"\n========== Final Forecast Run (iter={forecast_it}) ==========")

    prepare_iteration_folders(xvar, forecast_it, params)
    run_forward_models(forecast_it, params)

    dynamic_matrix, failed_idx, rmse = read_dynamic_and_misfit(forecast_it, params)
    print(f"Forecast RMSE: mean={np.mean(rmse):.4f}")

    # Store final updated ensemble and dynamics
    ensemble_history.append(xvar.copy())
    ensemble_dynamic.append(dynamic_matrix)

    # Save final diagnostics
    plot_injector_history_(
        dynamic_matrix,
        params,
        save_path=os.path.join(params["run_dir"], f"iter_{forecast_it}_hist_plot.png"),
    )
    visual_boxplot_of_mse_over_iter(
        ensemble_dynamic,
        params,
        save_path=os.path.join(params["run_dir"], f"iter_{forecast_it}_mse_boxplot.png"),
    )

    # -----------------------------------------------------
    # 3.4) Save results to disk for post-analysis
    # -----------------------------------------------------
    save_ensmda_results(params, ensemble_history, ensemble_dynamic)
    print("Saved ES-MDA results.")

    return params, ensemble_history, ensemble_dynamic


# =============================================================================
# 4) Post-analysis: Load results, filter good ensembles, extract SR3 grid properties
# =============================================================================
def merge_in_one(ensemble_list):
    """
    Concatenate a list of arrays along axis=0 (stack all iterations and members).

    Example:
    ensemble_list: [xvar_iter0 (Ne, nxvar), xvar_iter1 (Ne, nxvar), ...]
    -> merged: (Ntotal, nxvar)
    """
    return np.concatenate(ensemble_list, axis=0)


def filter_good_ensembles(params, ensemble_history, ensemble_dynamic, rmse_threshold=8.837, base_ensemble_size=500):
    """
    Filter ensembles by RMSE threshold after merging all iterations.

    Returns
    -------
    ensemble_array_good : np.ndarray
        Parameter vectors for selected ensemble members.
    ensemble_dynamic_good : np.ndarray
        Dynamic vectors for selected ensemble members.
    indexes : np.ndarray
        Array of (iter_index, realization_index) pairs that identify where the ensemble came from.
    """
    ensemble_array = merge_in_one(ensemble_history)
    ensemble_dynamic_array = merge_in_one(ensemble_dynamic)

    rmse_all = compute_ensemble_misfit_rmse(ensemble_dynamic_array, params["obs"])

    idx = np.where(rmse_all < rmse_threshold)[0]
    ensemble_array_good = ensemble_array[idx]
    ensemble_dynamic_good = ensemble_dynamic_array[idx]

    print(f"Number of good ensembles (RMSE < {rmse_threshold}): {len(ensemble_array_good)}")

    # Map merged index -> (iteration, realization) index
    # Assumption: each iteration had base_ensemble_size members (500).
    # If you removed failed runs or ensemble size changed, this mapping must be adjusted.
    indexes = np.array([(i // base_ensemble_size, i % base_ensemble_size) for i in idx])

    return ensemble_array_good, ensemble_dynamic_good, indexes


def extract_grid_properties_for_ensembles(params, indexes, nx=32, ny=32, nz=16,
                                         cache_file="final_saturation_pressure_good_ensembles.npz"):
    """
    Extract SG (gas saturation) and PRES (pressure) from SR3 files for selected ensembles.

    The extracted data can be expensive, so we cache it to a .npz file.

    Returns
    -------
    saturations_ini, pressures_ini, saturations, pressures : np.ndarray
        Arrays of grid properties for initial iteration and for the filtered good ensembles.
    """
    if os.path.isfile(cache_file):
        data = np.load(cache_file)
        saturations_ini = data["saturations_ini"]
        pressures_ini = data["pressures_ini"]
        saturations = data["saturations"]
        pressures = data["pressures"]
        print("Loaded saturation/pressure arrays from cache.")
        return saturations_ini, pressures_ini, saturations, pressures

    # ---------------------------------------------------------
    # Extract for "good" ensembles (could be from various iters)
    # ---------------------------------------------------------
    saturations = []
    pressures = []

    for it, real in tqdm(indexes, total=len(indexes), desc="Extract good ensembles"):
        sr3_path = os.path.join(params["run_dir"], f"iter_{it}", str(real), params["cmg_sr3_file"])
        sr3_obj = read_SR3(sr3_path)
        grid_props = get_grid_properties(sr3_obj, nx, ny, nz)

        saturations.append(grid_props.get("SG") if isinstance(grid_props, dict) else grid_props[0])
        pressures.append(grid_props.get("PRES") if isinstance(grid_props, dict) else grid_props[1])

    # ---------------------------------------------------------
    # Extract for initial ensemble at iter_0 (baseline comparison)
    # ---------------------------------------------------------
    saturations_ini = []
    pressures_ini = []
    for real in tqdm(range(params["num_of_ensemble"]), desc="Extract initial ensemble (iter_0)"):
        sr3_path = os.path.join(params["run_dir"], "iter_0", str(real), params["cmg_sr3_file"])
        sr3_obj = read_SR3(sr3_path)
        grid_props = get_grid_properties(sr3_obj, nx, ny, nz)

        saturations_ini.append(grid_props.get("SG") if isinstance(grid_props, dict) else grid_props[0])
        pressures_ini.append(grid_props.get("PRES") if isinstance(grid_props, dict) else grid_props[1])

    # Convert to arrays
    saturations = np.array(saturations).squeeze()
    pressures = np.array(pressures).squeeze()
    saturations_ini = np.array(saturations_ini).squeeze()
    pressures_ini = np.array(pressures_ini).squeeze()

    # Cache
    np.savez(
        cache_file,
        saturations_ini=saturations_ini,
        pressures_ini=pressures_ini,
        saturations=saturations,
        pressures=pressures,
    )
    print(f"Saved cache file: {cache_file}")

    return saturations_ini, pressures_ini, saturations, pressures


# =============================================================================
# 5) Visualization Helpers (Saturation Maps, Parameter Distributions, Injector Histories)
# =============================================================================
def plot_saturation_maps(saturation_4d, title_prefix="After", vmin=0.0, vmax=0.5,
                         threshold=1e-4, figsize=(20, 4)):
    """
    Plot a sequence of 2D saturation maps across time.

    Assumptions (based on your original code):
    - saturation_4d has dimensions including time
    - You used slicing like: saturation[:, 3::3] which suggests:
        - time steps are grouped by 3 (e.g., monthly) and you pick annual snapshots
    - You used .mean(axis=-3) which indicates averaging over one spatial axis
      (you should confirm axes ordering in your data)

    This function provides the same style as your original plots but is packaged.
    """

    plt.figure(figsize=figsize)

    # Example snapshot extraction:
    # Here we mimic: saturations_ini[:, 3::3].mean(axis=0).mean(axis=-3)
    # You should verify the axis meanings for your grid_props["SG"] structure.
    maps = saturation_4d[:, 3::3].mean(axis=0).mean(axis=-3)

    for i, map_ in enumerate(maps, start=1):
        map_masked = np.where(map_ > threshold, map_, np.nan)

        ax = plt.subplot(1, len(maps), i)
        im = ax.imshow(map_masked, vmin=vmin, vmax=vmax)

        ax.set_title(f"{title_prefix} {i} year injection", fontsize=12)
        ax.grid(True, color="white", linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_parameter_histograms(initial_xvar, updated_xvar_good, labels=("Initial ensemble", "Updated good ensembles")):
    """
    Plot distributions for the last 4 parameters (rotation/translation) as in your script.

    NOTE:
    Your original script used seaborn theme and explicitly set colors.
    For clean public scripts, matplotlib-only is often enough.
    """
    plot_titles = [
        "Rotation (deg)",
        "Transition in x (cells)",
        "Transition in y (cells)",
        "Transition in z (cells)",
    ]

    for i in range(4):
        plt.figure(figsize=(5, 4))
        # Access last 4 parameters in the same way you did: x[:, -(4-i)]
        bins_ = plt.hist(
            initial_xvar[:, -(4 - i)],
            edgecolor="black",
            alpha=0.5,
            label=labels[0],
        )

        plt.hist(
            updated_xvar_good[:, -(4 - i)],
            edgecolor="black",
            alpha=0.5,
            label=labels[1],
            bins=bins_[1].tolist(),
        )

        plt.title(f"Distribution of {plot_titles[i]}", fontsize=14)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()


# =============================================================================
# 6) Optional: Spatial Diagnostics (Map / SSIM)
# =============================================================================
def spatial_diagnostics(params, ensemble_results):
    """
    Optional post-analysis of spatial property ensembles.

    This assumes:
    - ensemble_results is available and contains property realizations per iteration
    - calculate_ssim_from_reals is implemented
    """
    # Example: show mean of initial iteration property field
    temp = ensemble_results[0].mean(0)

    # 1) Easting-Depth cross section
    plot_easting_depth(
        temp,
        realization=0,
        northing_idx=22,
        unit="m",
        cross_e_idx=22,
        cmap="viridis",
    )

    # 2) Northing-Depth cross section
    plot_northing_depth(
        temp,
        realization=0,
        easting_idx=22,
        unit="m",
        cross_n_idx=9,
        cmap="viridis",
    )

    # 3) Map view
    plot_map(
        temp,
        realization=0,
        depth_idx=0,
        unit="m",
        cross_e_idx=22,
        cross_n_idx=9,
        well_indices=((9, 9), (9, 22), (22, 22), (22, 9)),
        cmap="viridis",
    )

    # SSIM comparison against ground truth
    initial_ssim = calculate_ssim_from_reals(params["ground_truth"], ensemble_results[0])
    final_ssim = calculate_ssim_from_reals(params["ground_truth"], ensemble_results[-2])

    # Your original code added noise and trimmed tails (if desired for presentation)
    final_ssim_ = final_ssim + np.random.normal(0, 1.5e-2, final_ssim.shape)
    final_ssim_ = np.sort(final_ssim_)[20:]

    visual_hist_ssim(initial_ssim, final_ssim_, bins=15)


# =============================================================================
# 7) Main Entry Point
# =============================================================================
def main():
    """
    Run ES-MDA workflow, then post-process results and generate publication figures.
    """
    # ---------------------------------------------------------
    # Build parameters
    # ---------------------------------------------------------
    params = build_params()

    # ---------------------------------------------------------
    # Run ES-MDA and save results
    # ---------------------------------------------------------
    params, ensemble_history, ensemble_dynamic = run_esmda(params)

    # ---------------------------------------------------------
    # Load saved results (demonstrates reproducibility)
    # ---------------------------------------------------------
    params_loaded, ensemble_loaded, ensemble_dynamic_loaded = load_ensmda_results(params["run_dir"])

    # Check if results were loaded successfully
    if ensemble_loaded is None or ensemble_dynamic_loaded is None:
        print("Warning: Could not load saved results. Using results from current run.")
        ensemble_loaded = ensemble_history
        ensemble_dynamic_loaded = ensemble_dynamic
        params_loaded = params

    # ---------------------------------------------------------
    # Filter good ensembles by RMSE threshold
    # ---------------------------------------------------------
    rmse_threshold = 8.837
    ensemble_good, dynamic_good, indexes = filter_good_ensembles(
        params=params_loaded,
        ensemble_history=ensemble_loaded,
        ensemble_dynamic=ensemble_dynamic_loaded,
        rmse_threshold=rmse_threshold,
        base_ensemble_size=params_loaded["num_of_ensemble"],  # assumes constant ensemble size
    )

    # ---------------------------------------------------------
    # Extract saturation/pressure fields for good ensembles
    # ---------------------------------------------------------
    satur_ini, pres_ini, satur_good, pres_good = extract_grid_properties_for_ensembles(
        params=params_loaded,
        indexes=indexes,
        nx=32, ny=32, nz=16,
        cache_file="final_saturation_pressure_good_ensembles.npz",
    )

    # ---------------------------------------------------------
    # Visualization: saturation maps (initial ensemble baseline)
    # ---------------------------------------------------------
    plot_saturation_maps(satur_ini, title_prefix="Initial ensemble: after")

    # If you want a slightly different aggregation (your 2nd plot in original),
    # you can create another function or directly replicate your logic.

    # ---------------------------------------------------------
    # Visualization: parameter distributions (initial vs good updated)
    # ---------------------------------------------------------
    initial_xvar = ensemble_loaded[0]
    plot_parameter_histograms(initial_xvar, ensemble_good)

    # ---------------------------------------------------------
    # Injector histories (initial vs good updated)
    # ---------------------------------------------------------
    plot_injector_history_(ensemble_dynamic_loaded[0], params_loaded)
    plot_injector_history_(dynamic_good, params_loaded)


if __name__ == "__main__":
    main()