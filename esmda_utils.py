import os
import pickle
import numpy as np
from singan_generate import generate_sample
import shutil
from tqdm import tqdm
import pandas as pd
from cmg_sr3_reader import read_SR3, get_wells_timeseries, get_grid_properties
import glob
import numpy as np
from scipy.stats import norm

with open('params_latent_var.pkl', 'rb') as f:
    param_latent = pickle.load(f)
with open('params_sand_ratio_ground_truth.pkl', 'rb') as f:
    param_sand_ratio = pickle.load(f)
ground_truth = np.load('property_modelling_part/prop_ground_truth.npy')

# TODO: need to be pickled
param_mean_sand = {'lower_bound': 249.9, 'upper_bound': 250}
param_mean_shaly_sand =  {'lower_bound': 10, 'upper_bound': 10.1}
param_rotate_n_shift =  {'lower_rotate': -25., 
                         'upper_rotate': 25.,
                         'lower_translate_x': -8.0, 
                         'upper_translate_x': 8.0,
                         'lower_translate_y': -8.0, 
                         'upper_translate_y': 8.0,
                         'lower_translate_z': -8.0, 
                         'upper_translate_z': 8.0,}

start_date = pd.to_datetime("2025-01-01")
lst_well = ['Well 1', 'Well 2', 'Well 3', 'Well 4']
observed_data = ['BHP'] 
# observed_data = ['GASRATSC'] 

history_days = np.arange(15, 721, 15) # in days 
prediction_days = np.arange(15, 1801, 15) # in days 

def generate_x_vars(param_latent=param_latent, 
                    param_sand_ratio = param_sand_ratio,
                    param_mean_sand = param_mean_sand,
                    param_mean_shaly_sand = param_mean_shaly_sand,
                    random_seed=90102,
                    num_realizations=None, 
                    return_seed = False,
                    verbose = False,
                    no_latent = False):
    """
    Generate a set of random latent variables, vertical proportions, mean permeabilities for sand and shaly sand, and rotation and shift parameters.

    Parameters
    ----------
    param_latent : dict
        Parameters for the latent variables. Must contain 'num_of_latent' and 'nz'.
    param_sand_ratio : dict
        Parameters for the vertical proportions. Must contain 'lower_bound' and 'upper_bound'.
    param_mean_sand : dict
        Parameters for the mean permeability of sand. Must contain 'lower_bound' and 'upper_bound'.
    param_mean_shaly_sand : dict
        Parameters for the mean permeability of shaly sand. Must contain 'lower_bound' and 'upper_bound'.
    random_seed : int or None
        Random seed. If None, numpy's random seed will be used.
    num_realizations : int or None
        If not None, generate this number of realizations.
    return_seed : bool
        If True, return the random seed used for generating the realizations.
    verbose : bool
        If True, print some information about the generated realizations.

    Returns
    -------
    numpy array
        A numpy array of shape (num_realizations, ...) containing the generated realizations. The first column is the latent variables, the second column is the vertical proportions, the third column is the mean permeability of sand, the fourth column is the mean permeability of shaly sand, and the last columns are the rotation and shift parameters.
    """
    
    if verbose:
        print('random_seed:', random_seed)
    if num_realizations is not None and isinstance(num_realizations, int):
        ensemble = []
        for _ in range(num_realizations):
            real, random_seed = generate_x_vars(param_latent, 
                                         param_sand_ratio, 
                                         param_mean_sand,
                                         param_mean_shaly_sand,
                                         random_seed=random_seed, 
                                         return_seed=True)
            random_seed += 1
            ensemble.append(real)
        return np.array(ensemble)

    # Set seed
    if random_seed is not None:
        np.random.seed(random_seed)    
    # Generate random latent variables
    if no_latent:
        latent_vars = np.zeros(param_latent['num_of_latent'])
    else: 
        latent_vars = np.random.normal(0, 1, param_latent['num_of_latent'])
    # Generate vertical proportions
    vertical_ratios = []
    for i in range(param_latent['nz']):
        vertical_ratios.append(np.random.uniform(param_sand_ratio['lower_bound'][i], 
                                                 param_sand_ratio['upper_bound'][i]))
    vertical_ratios = np.array(vertical_ratios)
    # (outdated code) vertical_ratios = np.random.uniform(0.35, 0.75, size=param_latent['nz'])

    # Generate mean perms
    sand_perm = np.random.uniform(param_mean_sand['lower_bound'], param_mean_sand['upper_bound'])
    sand_perm = np.array(sand_perm).reshape(1,)
    shaly_sand_perm = np.random.uniform(param_mean_shaly_sand['lower_bound'], param_mean_shaly_sand['upper_bound'])
    shaly_sand_perm = np.array(shaly_sand_perm).reshape(1,)

    # Rotate and shift
    rotate = np.random.uniform(param_rotate_n_shift['lower_rotate'], 
                                       param_rotate_n_shift['upper_rotate'])
    rotate = np.array(rotate).reshape(1,)

    translate_x = np.random.uniform(param_rotate_n_shift['lower_translate_x'], 
                                  param_rotate_n_shift['upper_translate_x'])
    translate_x = np.array(translate_x).reshape(1,)

    translate_y = np.random.uniform(param_rotate_n_shift['lower_translate_y'],
                                  param_rotate_n_shift['upper_translate_y'])
    translate_y = np.array(translate_y).reshape(1,)

    translate_z = np.random.uniform(param_rotate_n_shift['lower_translate_z'],
                                  param_rotate_n_shift['upper_translate_z'])
    translate_z = np.array(translate_z).reshape(1,)

    if verbose:
        print('latent_vars:', latent_vars[:10])
        print('vertical_ratios:', vertical_ratios)
        print('sand_perm:', sand_perm)
        print('shaly_sand_perm:', shaly_sand_perm)
        print('rotate:', rotate)
        print('translate_x:', translate_x)
        print('translate_y:', translate_y)
        print('translate_z:', translate_z)

    # Generate mean rocktypes
    if return_seed:
        return np.concatenate((latent_vars, vertical_ratios, sand_perm, shaly_sand_perm, rotate, translate_x, translate_y, translate_z)), random_seed
    else:
        return np.concatenate((latent_vars, vertical_ratios, sand_perm, shaly_sand_perm, rotate, translate_x, translate_y, translate_z))
    


def generate_sample_from_xvar(xvar, param_latent=param_latent, no_latent=True):
    """
    Generate a 3D sample from SinGAN latent variables.

    Parameters
    ----------
    xvar : array-like, shape (n_realizations, n_latent_vars) or (n_latent_vars,)
        The SinGAN latent variables from which to generate a sample.
    param_latent : dict, default=param_latent
        The parameters for the latent variables.

    Returns
    -------
    ensemble : array-like, shape (n_realizations, nx, ny, nz)
        The generated ensemble of 3D samples.

    """
    
    if len(xvar.shape) == 2:
        ensemble = []
        for xvar_alone in tqdm(xvar, 'Generating samples from SinGAN latent variables...'):
            lst_latent, vertical_sand_portion_trend, rotate_n_shift  = split_xvar(xvar_alone, param_latent)
            if no_latent: 
                lst_latent = [np.zeros_like(i) for i in lst_latent]
            ensemble.append(generate_sample(lst_latent, 
                                            vertical_sand_portion_trend,
                                            rotate_n_shift))
        return np.array(ensemble)
    elif len(xvar.shape) == 1:
        lst_latent, vertical_sand_portion_trend, rotate_n_shift  = split_xvar(xvar, param_latent)
        print(rotate_n_shift)
        if no_latent: 
                lst_latent = np.zeros_like(lst_latent)
        return generate_sample(lst_latent, 
                               vertical_sand_portion_trend,
                               rotate_n_shift)
    else:
        raise ValueError("xvar should be 1D or 2D")
    



class NormalScoreTransformer:
    def __init__(self):
        # Store transformation tables for inverse transform
        self.vr_tables = []    # Sorted original values
        self.ns_tables = []    # Corresponding normal scores

    @staticmethod
    def _gauinv(p):
        """Inverse standard normal CDF."""
        return norm.ppf(p)

    @staticmethod
    def _gauss_cdf(x):
        """Standard normal CDF."""
        return norm.cdf(x)

    def fit_transform(self, arr):
        """
        Apply Normal Score Transform column-wise.

        Parameters
        ----------
        arr : np.ndarray
            Shape: (n_realizations, n_features)

        Returns
        -------
        ns_arr : np.ndarray
            Normal-score transformed array (same shape as input).
        """
        if arr.ndim != 2:
            raise ValueError("Input array must be 2D: (n_realizations, n_features)")

        n_real, n_feat = arr.shape
        ns_arr = np.zeros_like(arr, dtype=float)

        self.vr_tables = []
        self.ns_tables = []

        for col in range(n_feat):
            vr = arr[:, col]
            sorted_idx = np.argsort(vr)
            vr_sorted = vr[sorted_idx]

            # Equal weights
            ranks = np.arange(1, n_real + 1)
            probs = (ranks - 0.5) / n_real

            ns_values = self._gauinv(probs)

            # Store transformation table
            self.vr_tables.append(vr_sorted)
            self.ns_tables.append(ns_values)

            # Map each original value to its normal score
            ns_col = np.interp(vr, vr_sorted, ns_values)
            ns_arr[:, col] = ns_col

        return ns_arr

    def inverse_transform(self, ns_arr):
        """
        Inverse transform from normal scores to original scale.

        Parameters
        ----------
        ns_arr : np.ndarray
            Shape: (n_realizations, n_features), transformed data.

        Returns
        -------
        arr : np.ndarray
            Back-transformed array (same shape as input).
        """
        if ns_arr.shape[1] != len(self.vr_tables):
            raise ValueError("Number of features in input does not match fitted transformer.")

        arr = np.zeros_like(ns_arr, dtype=float)

        for col in range(ns_arr.shape[1]):
            ns_col = ns_arr[:, col]
            arr[:, col] = np.interp(ns_col, self.ns_tables[col], self.vr_tables[col])

        return arr



def _check_if_overwrite(filename_):
    if os.path.exists(filename_): 
        print(f"Warning: {filename_} already exists. Overwriting...")
def _write_to_file(filename_, ensemble):
    _check_if_overwrite(filename_)
    with open(filename_, 'wb') as f:
        pickle.dump(ensemble, f)

def load_ensmda_results(run_dir, filename='ensemble'):
    """
    Load ESMDA results saved by save_ensmda_results.

    Parameters
    ----------
    run_dir : str
        Path to the run directory where results were saved.
    filename : str, optional
        Base filename used when saving (default is 'ensemble').

    Returns
    -------
    params_esmda : dict
        Dictionary of ESMDA parameters.
    ensemble : object or None
        Loaded ensemble, if it exists, otherwise None.
    ensemble_dynamic : object or None
        Loaded dynamic ensemble, if it exists, otherwise None.
    """
    # load parameters
    params_file = os.path.join(run_dir, 'params_esmda.pkl')
    with open(params_file, 'rb') as f:
        params_esmda = pickle.load(f)

    # load ensemble if exists
    ensemble_file = os.path.join(run_dir, filename + '_xvar.pkl')
    ensemble = None
    if os.path.exists(ensemble_file):
        with open(ensemble_file, 'rb') as f:
            ensemble = pickle.load(f)

    # load dynamic ensemble if exists
    dynamic_file = os.path.join(run_dir, filename + '_dynamic.pkl')
    ensemble_dynamic = None
    if os.path.exists(dynamic_file):
        with open(dynamic_file, 'rb') as f:
            ensemble_dynamic = pickle.load(f)

    return params_esmda, ensemble, ensemble_dynamic
def save_ensmda_results(params_esmda,
                        ensemble = None,
                        ensemble_dynamic = None,
                        filename = 'ensemble'):
    
    filename_ = os.path.join(params_esmda['run_dir'], 'params_esmda.pkl')
    with open(filename_, 'wb') as f:
        pickle.dump(params_esmda, f)

    if ensemble is not None:
        filename_ = os.path.join(params_esmda['run_dir'], filename+'_xvar.pkl')
        # _write_to_file(filename, ensemble)
        with open(filename_, 'wb') as f:
            pickle.dump(ensemble, f)
    if ensemble_dynamic is not None:
        filename_ = os.path.join(params_esmda['run_dir'], filename+'_dynamic.pkl')
        # _write_to_file(filename_, ensemble_dynamic)
        with open(filename_, 'wb') as f:
            pickle.dump(ensemble_dynamic, f)
        

def delete_failed_ensemble(xvar, failed_idx:list[str]):
    if len(failed_idx) != 0:
        failed_idx_ = np.array([i.split('\\')[-1] for i in failed_idx]).astype(int)
        return np.delete(xvar, failed_idx_, axis=0)
    else:
        return xvar

def compute_ensemble_misfit_rmse(ensemble, ground_truth):
    misfit = []
    for real in ensemble:
        try:
            misfit_ = compute_misfit_rmse(real, ground_truth)
            misfit.append(misfit_)
        except:
            pass
    return np.array(misfit)


def compute_ensemble_misfit_mae(ensemble, ground_truth):
    """
    Computes the mean absolute error (MAE) misfit for each realization in the ensemble
    compared to the ground truth.

    Parameters
    ----------
    ensemble : list or np.ndarray
        A collection of realizations to compare against the ground truth.
    ground_truth : np.ndarray
        The ground truth data for comparison.

    Returns
    -------
    np.ndarray
        An array of MAE misfits for each realization in the ensemble.
    """
    misfit = []
    for real in ensemble:
        try:
            # Compute the MAE misfit for the current realization
            misfit_ = compute_misfit_mae(real, ground_truth)
            misfit.append(misfit_)
        except Exception as e:
            # Handle any exceptions during the misfit computation
            print(f"Warning: Misfit computation failed for a realization. Error: {e}")
            pass
    return np.array(misfit)

def compute_misfit_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))
def compute_misfit_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def compute_misfit_mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def compute_alpha_lst(N:int, r:float)->list[float]:
    alpha_raw = np.array([N/(r**(j)) for j in range(N)])
    return alpha_raw*np.sum(1/alpha_raw)

def read_ensemble_sr3_for_history_matching(
    sr3_file: str,
    target_dir: str,
    level_of_directory: int = 1,
    start_date: pd.Timestamp = start_date,
    wells: list[str] = lst_well,
    observed_data: list[str] = observed_data,
    history_days: np.ndarray = history_days,
    return_failed_reals = False,
) -> np.ndarray:

    folder_list = glob.glob(target_dir + '/*' * level_of_directory)
    ensemble_history = []
    failed_real_num = []
    for folder in tqdm(folder_list, desc='Reading SR3 files in ensemble'):
        sr3_path = os.path.join(folder, sr3_file)
        try:
            df = get_wells_timeseries(sr3_path)
            well_history = []
            for well in wells:
                data_history = []
                for data in observed_data:
                    df[well].index = start_date + pd.to_timedelta(df[well].index, unit='D')
                    x = df[well]['Days']
                    y = df[well][data].values
                    data_history.append(np.interp(history_days, x, y))
                well_history.append(data_history)
            ensemble_history.append(well_history)
        except:
            failed_real_num.append(folder.split('/')[-1])

    if return_failed_reals:
        return np.array(ensemble_history).reshape(
            len(folder_list)-len(failed_real_num), len(wells), len(observed_data), len(history_days)
        ), failed_real_num
    else:
        return np.array(ensemble_history).reshape(
            len(folder_list)-len(failed_real_num), len(wells), len(observed_data), len(history_days)
        )



def read_ensemble_sr3(
    sr3_file: str,
    target_dir: str,
    level_of_directory: int = 1,
    ) -> np.ndarray:
    folder_list = glob.glob(target_dir + '/*' * level_of_directory)
    ensemble = []
    for folder in tqdm(folder_list, desc='Reading SR3 files in ensemble'):
        sr3_path = os.path.join(folder, sr3_file)
        ensemble.append(get_wells_timeseries(sr3_path))
    return np.array(ensemble)
        
def copy_files(files, target_dir: str | list[str], verbose: bool = False):
    """
    Copies given files to the target directory.

    Args:
        files (str or list[str]): A file or a list of files to be copied.
        target_dir (str or list[str]): The target directory where the files will be copied.
        verbose (bool, optional): If True, prints a success message. Defaults to False.
    """

    # Check if target directory is a list
    if isinstance(target_dir, list):
        for dir in tqdm(target_dir, 'Copying files to multiple directories...'):
            copy_files(files, dir)
        return

    # Check if target directory exists
    if not os.path.exists(target_dir): 
        os.makedirs(target_dir, exist_ok=True)
        if verbose: print(f"Created {target_dir}")

    # Copy the files
    if isinstance(files, list):
        # If files is a list, loop through each item and copy it
        for file in files:
            copy_files(file, target_dir)
    elif isinstance(files, str):
        # If files is a str, copy it to the target directory
        target_file = os.path.join(target_dir, os.path.basename(files))
        if not os.path.exists(target_file): shutil.copy(files, target_file)
    else:
        # If files is of any other type, raise an error
        raise TypeError(f"Unsupported type {type(files)}")

def replace_perm_and_rocktype(xvar, target_dirs, ground_truth=ground_truth):
    """
    Generates samples from xvar, assigns hard data and replaces the rocktype and permeability files in the target directories.

    Args:
        xvar (ndarray): The input latent variable.
        target_dirs (str or list[str]): The target directories where the rocktype and permeability files need to be replaced.
        ground_truth (ndarray, optional): The ground truth data. Defaults to the `ground_truth` variable defined in the script.
    """
    if isinstance(target_dirs, str):
        target_dirs = [target_dirs]
    for dir in target_dirs:
        if not os.path.exists(dir):
            raise AssertionError(f"Folder {dir} does not exist")
    if xvar.shape[0] != len(target_dirs):
        raise AssertionError(f"Number of target folders ({len(target_dirs)}) doesn't match number of realizations ({xvar.shape[0]})")

    # Generate sample from xvar
    realizations = generate_sample_from_xvar(xvar)
    mean_perm = form_sand_n_shaly_sand_perm(xvar)
    # Assign hard data
    if ground_truth is not None:
        # Assign hard data at the four corners
        hard_data_indices = [(9, 9), (22, 9), (9, 22), (22, 22)] # inj-1, 2, 3, 4 sequentially in the array
        for idx in hard_data_indices:
            realizations[:, :, idx[0], idx[1]] = ground_truth[:, idx[0], idx[1]]

    # Convert realizations to rocktype and permeability
    rocktype = (realizations + 2).astype(int)  # sand: 3; shaly-sand: 2
    permeability = []
    for i, real in enumerate(realizations):
        permeability.append(real*mean_perm[i,0] + (1-real)*mean_perm[i,1])  # 10(sand)-250(shaly)

    # Save the rocktype and permeability files
    filenames = ['rocktype.inc', 'permeability.inc']
    arrays = [rocktype, permeability]
    

    for i, target_dir in tqdm(enumerate(target_dirs), desc="Replacing rocktype and permeability files w/ sinGAN reals"):
        for arr, filename in zip(arrays, filenames):
            filepath = os.path.join(target_dir, filename)
            # Read first line (header)
            with open(filepath, "r") as f:
                header = f.readline()
            if 'rocktype' in filename:
                # Overwrite file, keeping first line
                with open(filepath, "w") as f:
                    f.write(header)                         # Keep original header
                    np.savetxt(f, arr[i].flatten(), fmt="%d")  # Write array below header
            else:
                # Overwrite file, keeping first line
                with open(filepath, "w") as f:
                    f.write(header)                             # Keep original header
                    np.savetxt(f, arr[i].flatten(), fmt="%.2f")  # Write array below header

def form_lst_latent(xvar_latent, param_latent=param_latent, verbose = False):
    # generate random latent variables:
    lst_latent = []
    former_num = 0
    for dim, num in zip(param_latent['lst_of_dim_latent'], 
                        param_latent['lst_of_num_latent']):
        latent = xvar_latent[former_num:former_num+num].reshape(dim).astype('float32')
        lst_latent.append(latent)
        if verbose:
            print(latent.shape)
        former_num = num
    return lst_latent
def form_vertical_sand_portion_trend(xvar, param_latent=param_latent, verbose = False):
    return xvar[param_latent['num_of_latent']:param_latent['num_of_latent']+param_latent['nz']]

def form_sand_n_shaly_sand_perm(xvar, verbose=False):
    xvar = np.array(xvar, dtype=float)  # ensure NumPy array
    
    if xvar.ndim == 1:  # single sample
        mean_perm = xvar[param_latent['num_of_latent']+param_latent['nz']:param_latent['num_of_latent']+param_latent['nz']+2].copy()
        mean_perm[0] = np.clip(mean_perm[0],
                               param_mean_sand['lower_bound'],
                               param_mean_sand['upper_bound'])
        mean_perm[1] = np.clip(mean_perm[1],
                               param_mean_shaly_sand['lower_bound'],
                               param_mean_shaly_sand['upper_bound'])
        if verbose:
            print("Clipped single sample:", mean_perm)
        return mean_perm
    
    elif xvar.ndim >= 2:  # multiple samples
        mean_perm = xvar[:, param_latent['num_of_latent']+param_latent['nz']:param_latent['num_of_latent']+param_latent['nz']+2].copy()
        mean_perm[:, 0] = np.clip(mean_perm[:, 0],
                                  param_mean_sand['lower_bound'],
                                  param_mean_sand['upper_bound'])
        mean_perm[:, 1] = np.clip(mean_perm[:, 1],
                                  param_mean_shaly_sand['lower_bound'],
                                  param_mean_shaly_sand['upper_bound'])
        if verbose:
            print("Clipped multiple samples:\n", mean_perm)
        return mean_perm
    
    else:
        raise ValueError("Input xvar must be at least 1D.")

def form_rotate_n_shift(xvar, verbose=False):
    xvar = np.array(xvar, dtype=float)  # ensure NumPy array

    if xvar.ndim == 1:  # single sample
        rotate_n_shift = xvar[-4:].copy()
        rotate_n_shift[0] = np.clip(rotate_n_shift[0], 
                                    param_rotate_n_shift['lower_rotate'],
                                    param_rotate_n_shift['upper_rotate'])
        rotate_n_shift[1] = np.clip(rotate_n_shift[1], 
                                    param_rotate_n_shift['lower_translate_x'],
                                    param_rotate_n_shift['upper_translate_x'])
        rotate_n_shift[2] = np.clip(rotate_n_shift[2], 
                                    param_rotate_n_shift['lower_translate_y'],
                                    param_rotate_n_shift['upper_translate_y'])
        rotate_n_shift[3] = np.clip(rotate_n_shift[3], 
                                    param_rotate_n_shift['lower_translate_z'],
                                    param_rotate_n_shift['upper_translate_z'])
        if verbose:
            print("rotate_n_shift single sample:", rotate_n_shift)
        return rotate_n_shift
    
    elif xvar.ndim >= 2:  # multiple samples
        rotate_n_shift = xvar[:, -4:].copy()
        
        rotate_n_shift[:,0] = np.clip(rotate_n_shift[:,0], 
                                      param_rotate_n_shift['lower_rotate'],
                                      param_rotate_n_shift['upper_rotate'])
        rotate_n_shift[:,1] = np.clip(rotate_n_shift[:,1], 
                                      param_rotate_n_shift['lower_translate_x'],
                                      param_rotate_n_shift['upper_translate_x'])
        rotate_n_shift[:,2] = np.clip(rotate_n_shift[:,2], 
                                      param_rotate_n_shift['lower_translate_y'],
                                      param_rotate_n_shift['upper_translate_y'])
        rotate_n_shift[:,3] = np.clip(rotate_n_shift[:,3], 
                                      param_rotate_n_shift['lower_translate_z'],
                                      param_rotate_n_shift['upper_translate_z'])
        if verbose:
            print("rotate_n_shift multiple samples:\n", rotate_n_shift)
        return rotate_n_shift
    
    else:
        raise ValueError("Input xvar must be at least 1D.")

def split_xvar(xvar_latent, param_latent=param_latent, verbose = False):
    return (form_lst_latent(xvar_latent, param_latent, verbose), 
            form_vertical_sand_portion_trend(xvar_latent, param_latent),
            form_rotate_n_shift(xvar_latent, verbose))
