import matplotlib.pyplot as plt
from esmda_utils import compute_ensemble_misfit_rmse
import numpy as np
import pyvista as pv

def visual_3d_subplots_3d(properties, 
                          nrows=4, 
                          ncols=4, 
                          aspect_x_to_z=1, cmap='viridis', is_xyz_seq=False):
    """
    Visualize multiple 3D properties in a grid layout as 3D subplots with flipped Z-axis.

    Args:
        properties (list of np.ndarray): List of 3D property arrays to visualize.
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.
        aspect_x_to_z (float): Aspect ratio for the z-direction scaling.
        cmap (str): Colormap for visualization.
        is_xyz_seq (bool): If True, transpose the property before visualization.

    Returns:
        None
    """
    # Create a PyVista Multi-Window Plotter with no borders
    plotter = pv.Plotter(shape=(nrows, ncols), border=False)      
    plotter.set_background("white") 
    for i, property in enumerate(properties):
        if i >= nrows * ncols:
            break

        row, col = divmod(i, ncols)
        plotter.subplot(row, col)

        # Apply GSLIB -> PyVista transformation
        if is_xyz_seq:
            property = property.T
        property = property[::-1, ::-1, ::-1]  
        # Create PyVista grid
        grid = pv.ImageData()
        nz, ny, nx = property.shape
        grid.dimensions = np.array([nx, ny, nz]) + 1
        grid.origin = (0, 0, 0)
        grid.spacing = (1, 1, aspect_x_to_z)
        grid.cell_data['values'] = property.flatten()

        # Add the 3D property to the subplot (No edges, No colorbar, No black borders)
        plotter.add_mesh(grid, cmap=cmap, show_edges=False, show_scalar_bar=False)

    plotter.show()

# viz_sections.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FT_TO_M = 0.3048

def make_custom_cividis():
    cividis = plt.get_cmap('cividis')
    colors = [cividis(0), cividis(0.6)]
    return LinearSegmentedColormap.from_list('custom_cividis', colors)

def _common_ticks(F, depth_ft_range=(3022, 3570), horiz_ft=(0, 18000), unit="m"):
    """
    원래 코드의 tick 계산을 한 번에 수행해서 넘겨줌.
    F는 (R,D,N,E) 또는 (D,N,E) 구조 모두 허용.
    """
    arr = F if F.ndim == 3 else F[0]
    ny, nn, ne = arr.shape  # (depth, northing, easting)

    # 공통 축 위치
    tick_positions = np.linspace(0, ne-1, 6)   # easting/northing 모두 6개 눈금

    # 단위 변환
    def conv(x): return np.asarray(x) * FT_TO_M if unit == "m" else np.asarray(x)

    # x 라벨 (Easting: 0→18000), y 라벨(Depth)
    tick_labels1 = conv(np.linspace(horiz_ft[0], horiz_ft[1], 6)).astype(int)
    tick_labels2 = conv(np.linspace(horiz_ft[1], horiz_ft[0], 6)).astype(int)
    depth_ticks = conv(np.linspace(*depth_ft_range, ny))
    tick_indices = range(0, ny, 4)

    return tick_positions, tick_labels1, tick_labels2, depth_ticks, tick_indices

def plot_easting_depth(F, realization=0, northing_idx=22,
                       depth_ft_range=(3022, 3570), horiz_ft=(0, 18000),
                       unit="m", cross_e_idx=22, cmap=None, figsize=(5,5)):
    """Easting vs Depth 단면 (원래 코드 로직 그대로)."""
    if cmap is None: cmap = make_custom_cividis()
    arr = F if F.ndim == 3 else F[realization]
    tick_positions, tick_labels1, _, depth_ticks, tick_indices = _common_ticks(
        F, depth_ft_range, horiz_ft, unit
    )

    plt.figure(figsize=figsize)
    plt.imshow(arr[:, northing_idx, :], cmap=cmap)

    plt.xlabel(f'Easting, {unit}', fontsize=10)
    plt.ylabel(f'Layers (0 - top, 15 - bottom)', fontsize=10)
    plt.xticks(tick_positions, tick_labels1, fontsize=10)
    plt.yticks([0,5,10,15],[0,5,10,15], fontsize=10)
    # plt.yticks(tick_indices, np.round(depth_ticks[tick_indices]).astype(int), fontsize=10)

    if cross_e_idx is not None:
        plt.axvline(x=cross_e_idx, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

def plot_northing_depth(F, realization=0, easting_idx=22,
                        depth_ft_range=(3022, 3570), horiz_ft=(0, 18000),
                        unit="m", cross_n_idx=9, cmap=None, figsize=(5,5)):
    """Northing vs Depth 단면 (flip + top ticks, 원 코드 그대로)."""
    if cmap is None: cmap = make_custom_cividis()
    arr = F if F.ndim == 3 else F[realization]
    tick_positions, _, tick_labels2, depth_ticks, tick_indices = _common_ticks(
        F, depth_ft_range, horiz_ft, unit
    )

    plt.figure(figsize=figsize)
    plt.imshow(np.flip(arr[:, :, easting_idx]), cmap=cmap)

    plt.title(f'Northing, {unit}', fontsize=10)
    plt.ylabel(f'Layers (0 - top, 15 - bottom)', fontsize=10)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xticks(tick_positions, tick_labels2, fontsize=10, rotation=90)
    plt.yticks([0,5,10,15],[15,10,5,0], fontsize=10)

    # plt.yticks(tick_indices, np.round(depth_ticks[tick_indices]).astype(int),
    #            fontsize=10, rotation=90)


    if cross_n_idx is not None:
        plt.axvline(x=cross_n_idx, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

def plot_map(F, realization=0, depth_idx=0,
             horiz_ft=(0, 18000), unit="ft",
             cross_e_idx=22, 
             cross_n_idx=9,
             well_indices=((9,9),
                           (9,22),
                           (22,22),
                           (22,9)),
             cmap=None, figsize=(5,5)):
    """Easting vs Northing 평면도 (flip + top x, y 라벨 역방향)."""
    if cmap is None: cmap = make_custom_cividis()
    arr = F if F.ndim == 3 else F[realization]
    tick_positions, tick_labels1, tick_labels2, _, _ = _common_ticks(
        F, depth_ft_range=(0,1), horiz_ft=horiz_ft, unit=unit
    )

    plt.figure(figsize=figsize)
    plt.imshow(np.flip(arr[depth_idx, :], axis=0), cmap=cmap)

    plt.title(f'Easting, {unit}', fontsize=10)
    plt.ylabel(f'Northing, {unit}', fontsize=10)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    plt.xticks(tick_positions, tick_labels1, fontsize=10)
    plt.yticks(tick_positions, tick_labels2, fontsize=10)

    if cross_e_idx is not None:
        plt.axvline(x=cross_e_idx, color='red', linestyle='--', label='Cross Section')
    if cross_n_idx is not None:
        plt.axhline(y=cross_n_idx, color='red', linestyle='--')

    if well_indices:
        for (n_i, e_i) in well_indices:
            plt.scatter(e_i, n_i, facecolors='none', edgecolors='black',
                        s=100, linewidth=1.5, marker='^', label='Well Location')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()

def visualize_3d_property(
    data: np.ndarray,
    aspect_z: float = 1.0,
    opacity: float = 1.0,
    show_edges: bool = False,
    cmap: str = "viridis",
    scalar_name: str = "value",
    threshold: float | None = None,
    clim: tuple[float, float] | None = None,
    is_xyz_order: bool = False,
    cut_i: int | None = None,
    cut_j: int | None = None,
    cut_k: int | None = None,
    show_scalar_bar: bool = False,
    filepath: str = None
) -> None:
    """
    Visualize a 3D property array using PyVista, with optional cutting along i, j, k indices.

    Args:
        data (np.ndarray): 3D property array to visualize.
        aspect_z (float, optional): Scaling factor for the z-axis spacing. Defaults to 1.0.
        show_edges (bool, optional): Whether to display mesh cell edges. Defaults to False.
        cmap (str, optional): Colormap for visualization. Defaults to 'cividis'.
        scalar_name (str, optional): Name of the scalar field for coloring. Defaults to 'value'.
        threshold (float, optional): Threshold value. If provided, only cells above this value are shown. Defaults to None.
        clim (tuple[float, float], optional): Color scale limits (min, max). Defaults to None.
        is_xyz_order (bool, optional): If True, assumes input array is ordered as (x, y, z).
        cut_i (int, optional): Index along x-axis (i-direction) to slice.
        cut_j (int, optional): Index along y-axis (j-direction) to slice.
        cut_k (int, optional): Index along z-axis (k-direction) to slice.

    Returns:
        None
    """
    if is_xyz_order:
        data = data.T

    # Convert GSLIB convention to PyVista convention
    data = data[::-1, ::-1, ::-1]
    nz, ny, nx = data.shape

    # Build grid
    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, aspect_z)
    grid.cell_data[scalar_name] = data.ravel(order="C")

    if threshold is not None:
        grid = grid.threshold(threshold)

    plotter = pv.Plotter()

    # Add full 3D mesh
    plotter.add_mesh(grid, show_edges=show_edges, 
                     cmap=cmap, clim=clim, 
                     opacity=opacity,
                     show_scalar_bar=show_scalar_bar)

    # Add slices at requested indices
    if cut_i is not None:  # X direction slice
        x_coord = cut_i
        slice_x = grid.slice(normal=(1, 0, 0), origin=(x_coord, 0, 0))
        plotter.add_mesh(slice_x, cmap=cmap, clim=clim, show_edges=False)

    if cut_j is not None:  # Y direction slice
        y_coord = cut_j
        slice_y = grid.slice(normal=(0, 1, 0), origin=(0, y_coord, 0))
        plotter.add_mesh(slice_y, cmap=cmap, clim=clim, show_edges=False)

    if cut_k is not None:  # Z direction slice
        z_coord = cut_k * aspect_z
        slice_z = grid.slice(normal=(0, 0, 1), origin=(0, 0, z_coord))
        plotter.add_mesh(slice_z, cmap=cmap, clim=clim, show_edges=False)

    
    if filepath is not None:
        plotter.show(screenshot=filepath)  # Save screenshot
    else:
        plotter.show()  # Interactive view



def visual_update_of_i_xvar(ensemble, i_xvar: int | list[int], save_path=None):
    """
    Visualize the distribution of selected variable(s) across ensemble iterations.

    Parameters
    ----------
    ensemble : list of np.ndarray
        Each element = (n_samples, n_variables) for a given iteration.
    i_xvar : int | list[int]
        Index (or indices) of variable(s) to visualize.
    save_path : str, optional
        If provided, saves the figure instead of showing it.
    """
    def _plot_single(ax, i_xvar):
        data = [en_[:, i_xvar] for en_ in ensemble]

        # Boxplot with style
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", color="blue", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2),
                   whiskerprops=dict(color="blue"),
                   capprops=dict(color="blue"))

        # Overlay jittered scatter points
        for i, vals in enumerate(data, start=1):
            x = np.random.normal(i, 0.05, size=len(vals))  # jitter
            ax.scatter(x, vals, alpha=0.5, s=20, color="gray")

        # Labels
        ax.set_xticks(range(1, len(data)+1))
        ax.set_xticklabels([f"Iter {i}" for i in range(1, len(data)+1)])
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"Xvar {i_xvar} Value")
        ax.set_title(f"Distribution of Xvar {i_xvar}")

        ax.grid(alpha=0.3)

    # Single variable
    if isinstance(i_xvar, int):
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_single(ax, i_xvar)

    # Multiple variables
    elif isinstance(i_xvar, list):
        n_vars = len(i_xvar)
        n_cols = min(n_vars, 3)
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = np.array(axes).reshape(-1)  # flatten in case of 1D

        for ax, var_idx in zip(axes, i_xvar):
            _plot_single(ax, var_idx)

        # Hide unused subplots if any
        for ax in axes[len(i_xvar):]:
            ax.axis("off")

        plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
def visual_boxplot_of_mse_over_iter(ensemble_dynamic, params_esmda, save_path=None):
    # Compute RMSE for each iteration
    """
    Visualize the misfit (RMSE) for each iteration of an ESDA run.

    Parameters
    ----------
    ensemble_dynamic : list of ndarrays
        A list of ndarrays, each containing the dynamic variables for an iteration.
    params_esmda : dict
        A dictionary containing the parameters for the ESDA run.
    save_path : str or None
        The path to save the figure. If None, the figure is displayed.

    Returns
    -------
    None
    """

    rmse_ensemble = [
        compute_ensemble_misfit_rmse(dynamic, params_esmda['obs'])
        for dynamic in ensemble_dynamic
    ]

    # Create boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(
        rmse_ensemble,
        patch_artist=True,
        labels=[f"Iter {i+1}" for i in range(len(rmse_ensemble))]
    )

    # Style boxplot
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(rmse_ensemble)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)

    # Add scatter points for medians
    for i, rmse in enumerate(rmse_ensemble, start=1):
        ax.scatter(
            np.repeat(i, len(rmse)),
            rmse,
            alpha=0.4,
            color='black',
            s=15
        )

    # Labels & grid
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Misfit (RMSE)", fontsize=12)
    ax.set_title("Ensemble Misfit RMSE by Iteration", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def visual_hist_ssim(ensemble, final = None, select=  None, bins = 20):
    plt.figure(figsize=(6, 4))

    plt.hist(ensemble, bins=bins, color='blue', alpha=0.5, edgecolor='black', label='Iniital ensemble', density = False)
    if final is not None: plt.hist(final,   bins=bins, color='red',  alpha=0.5, edgecolor='black', label='Updated ensemble', density = False)
    if select is not None: plt.hist(select,   bins=bins, color='green',  alpha=0.5, edgecolor='black', label='Final Ensemble', density = False)

    plt.xlabel("Structural similarity index metric [0-1] ", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("SSIM Distribution", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
def visual_hist_of_mean_perm(ensemble,iter):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Mean sandstone permeability
    axes[0].set_title("Mean Sandstone Permeability")
    axes[0].hist(ensemble[iter][:, -(2+4)], bins=20, color='skyblue', edgecolor='black', label='Ensemble')
    axes[0].axvline(x=250, color='red', linestyle='--', linewidth=2, label='Ground Truth')
    axes[0].set_xlabel("Permeability, mD")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Mean shaly sandstone permeability
    axes[1].set_title("Mean Shaly Sandstone Permeability")
    axes[1].hist(ensemble[iter][:, -(1+4)], bins=20, color='lightgreen', edgecolor='black', label='Ensemble')
    axes[1].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Ground Truth')
    axes[1].set_xlabel("Permeability, mD")
    axes[1].legend()

    fig.tight_layout()
    plt.show()

def visual_vertical_proportion_trend(ensembles, params_esmda, iteration, is_last=False):
    """
    Plot the vertical proportion trends for a given ensemble iteration, 
    including P10/P50/P90 percentiles, the ground truth, 
    and optionally the final ensemble.

    Parameters
    ----------
    ensembles : list of np.ndarray
        List of ensemble arrays, each of shape (n_realizations, n_features).
        The last 18 to last 2 columns (-18:-2) contain vertical proportion data.
    params_esmda : dict
        Must contain:
            'ground_truth_trend' : array-like, shape (16,)
                The reference vertical proportion trend.
    iteration : int
        Index of the current iteration to visualize.
    is_last : bool, default=False
        If True, also plots the last ensemble in blue.
    """
    
    depth_levels = np.arange(16)  # Assuming 16 depth layers
    data_current = ensembles[iteration][:, -(18+4):-(2+4)]  # Shape: (n_realizations, 16)

    # Percentile calculations
    p10 = np.percentile(data_current, 10, axis=0)
    p50 = np.percentile(data_current, 50, axis=0)
    p90 = np.percentile(data_current, 90, axis=0)

    # Plot all realizations for current iteration (light gray)
    plt.plot(data_current.T, depth_levels, color='gray', alpha=0.01)

    # Plot percentiles
    plt.plot(p10, depth_levels, '--b', label='P10 & P90')
    plt.plot(p90, depth_levels, '--b')
    plt.plot(p50, depth_levels, '-b', label='P50')

    # Plot ground truth
    plt.plot(params_esmda['ground_truth_trend'], depth_levels,
             color='red', linewidth=2, label='Ground Truth')

    # Optionally plot last ensemble
    if is_last:
        data_last = ensembles[-1][:, -18:-2]
        plt.plot(data_last.T, depth_levels, color='blue', alpha=0.1)
    
    # Axis settings
    plt.gca().invert_yaxis()
    plt.xlabel("Proportion")
    plt.ylabel("Depth Layer")
    plt.title(f"Vertical Proportion Trend (MAE = {np.mean(np.abs(p50 - params_esmda['ground_truth_trend'])):.3f})")

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
def plot_injector_history_(dynamic_matrix,
                           params_esmda, 
                           ncols=2, 
                           save_path=None):
    obs = np.squeeze(params_esmda['obs'])
    dynamic_matrix = np.squeeze(dynamic_matrix.T)
    
    n_wells = obs.shape[0]
    nrows = int(np.ceil(n_wells / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(n_wells):
        ax = axes[i]
        ax.plot(dynamic_matrix[:, i], color='gray', alpha=0.4)
        ax.plot(obs[i], color='red', linewidth=2, label='Observed')
        rmse = compute_ensemble_misfit_rmse(dynamic_matrix[:, i].T, obs[i])
        ax.set_title(f'Well {i+1} (rmse = {rmse.mean():.2f})', fontsize=8)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Hide unused subplots if n_wells is not a multiple of ncols
    for j in range(n_wells, len(axes)):
        axes[j].axis('off')
    
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()



def plot_injector_future_only_gt(
                         ground_truth, 
                         best_guess, 
                         params_esmda,
                         ncols=2, 
                         save_path=None,
                         days=None,
                         hist_days=None,
                         ):
    """
    Academic-style visualization of injector forecasts vs. observed and ground truth data.
    """
    from matplotlib import rcParams

    # Use LaTeX-style fonts and improve overall aesthetics
    rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    future = np.squeeze(ground_truth)
    best_guess = np.squeeze(best_guess)
    obs = np.squeeze(params_esmda['obs'])
    
    n_wells = obs.shape[0]
    nrows = int(np.ceil(n_wells / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(n_wells):
        ax = axes[i]

        # Ensemble simulations
        if days is not None and hist_days is not None:
            ax.plot(days, best_guess[i], color='green', linewidth=2, label='Training datum model', zorder=3)

            ax.plot(days, future[i], color='navy', linewidth=2, label='Reference operation condition', zorder=3)
            ax.plot(hist_days, future[i,:obs[i].shape[0]], color='darkred', linewidth=2, marker='o',
                    markersize=3, label='Observed period', zorder=3)

            ax.set_title('Injector {}'.format(i+1), fontsize=12, fontweight='bold')
        else:
            ax.plot(future[i], color='navy', linewidth=2, label='Ground Truth', zorder=3)
            ax.plot(future[i,:obs[i].shape[0]], color='darkred', linewidth=2, marker='o',
                    markersize=2, label='Observed', zorder=2)

        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel("Time (days)" if days is not None else "Index")
        # ax.set_ylabel("Injection rate (scf/day)")
        ax.set_ylabel("Flowing BHP (psi)")

    # Hide unused subplots
    for j in range(n_wells, len(axes)):
        axes[j].axis("off")

    # Global legend instead of repeating in every subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for global legend

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_injector_future(dynamic_matrix,
                         ground_truth, 
                         params_esmda,
                         ncols=2, 
                         save_path=None,
                         days=None,
                         hist_days=None,
                         ylim=[1650, 2100]):
    """
    Academic-style visualization of injector forecasts vs. observed and ground truth data.
    """
    from matplotlib import rcParams

    # Use LaTeX-style fonts and improve overall aesthetics
    rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    future = np.squeeze(ground_truth)
    obs = np.squeeze(params_esmda['obs'])
    dynamic_matrix = np.squeeze(dynamic_matrix.T)
    
    n_wells = obs.shape[0]
    nrows = int(np.ceil(n_wells / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(n_wells):
        ax = axes[i]

        # Ensemble simulations
        if days is not None and hist_days is not None:
            ax.plot(days, dynamic_matrix[:, i], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
            ax.plot(days[obs[i].shape[0]:], future[i,obs[i].shape[0]:], color='navy', linewidth=2, label='Ground Truth', zorder=3)
            ax.plot(hist_days, obs[i], color='darkred', linewidth=1, marker='o',
                    markersize=3, label='Observed', zorder=2)
        else:
            ax.plot(dynamic_matrix[:, i], color='gray', alpha=0.3, linewidth=0.8, zorder=1)
            ax.plot(future[i], color='navy', linewidth=2, label='Ground Truth', zorder=3)
            ax.plot(obs[i], color='darkred', linewidth=2, marker='o',
                    markersize=3, label='Observed', zorder=2)

        # Compute RMSE (optional, check dimensions)
        try:
            rmse = compute_ensemble_misfit_rmse(dynamic_matrix[:obs[i].shape[0], i].T, obs[i])
            ax.set_title(f'Injector {i+1}  |  RMSE = {rmse.mean():.2f}', fontsize=11)
        except Exception:
            ax.set_title(f'Injector {i+1}', fontsize=11)

        ax.grid(True, linestyle='--', alpha=0.4)
        # ax.set_ylim(ylim)
        ax.set_xlabel("Time (days)" if days is not None else "Index")
        ax.set_ylabel("Flowing BHP (psi)")

    # Hide unused subplots
    for j in range(n_wells, len(axes)):
        axes[j].axis("off")

    # Global legend instead of repeating in every subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for global legend

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_injector_history(dynamic_matrix, measured_obs):
    """
    Plots simulated ensemble and measured observations for injectors.
    
    Parameters
    ----------
    dynamic_matrix : np.ndarray
        Shape (n_time, n_wells, n_ensemble), simulated values.
    measured_obs : np.ndarray
        Shape (n_time, n_wells, n_obs_per_well), measured values.
    """
    n_wells = dynamic_matrix.shape[1]
    nrows = int(np.ceil(n_wells / 2))
    ncols = 2 if n_wells > 1 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 + 2 * (nrows - 1)), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for well_i, ax in enumerate(axes[:n_wells]):
        # Simulated ensemble
        ax.plot(dynamic_matrix[:, well_i, :].squeeze().T, color='gray', alpha=0.3, linewidth=0.8)

        # Measured observations
        ax.plot(measured_obs[:, well_i, :].squeeze(), 'r.', markersize=6, label='Measured')

        ax.set_title(f'Injector {well_i + 1}', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

        if well_i == 0:
            ax.legend()

    fig.text(0.5, 0.04, 'Time step', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Observation value', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.show()
