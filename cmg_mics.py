from datetime import datetime
import pickle
import os
import glob

def save_xvars(ensemble_dir, realization, file_name = None):

    """
    Saves the XVars realization to a pickle file in the ensemble directory.

    Parameters
    ----------
    ensemble_dir : str
        The location of the ensemble directory.
    realization : dict
        A dictionary of the XVars realization.
    file_name : str, optional
        The name of the file to save the XVars realization to. If not provided, a default name will be used.

    Returns
    -------
    None
    """
    ensemble_dir=_check_iterative_n_replace(ensemble_dir)

    if file_name is None:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = f"XVars_{current_time}.pkl"

    # Pickle the dictionary to the file
    with open(os.path.join(ensemble_dir, file_name), 'wb') as file:
        pickle.dump(realization, file)

        
def save_wells_dynamic_output(ensemble_dir, wells_data_ensemble, file_name = None):
    """
    Saves the wells data ensemble to a pickle file in the ensemble directory.

    Parameters
    ----------
    ensemble_dir : str
        The location of the ensemble directory.
    wells_data_ensemble : list
        A list of dictionaries, where each dictionary contains the data for a single well
    file_name : str, optional
        The name of the file to save the data to. If not provided, a default name will be used.

    Returns
    -------
    None
    """
    iter_dir = glob.glob(ensemble_dir + '/iter*')
    if iter_dir:
        ensemble_dir = iter_dir[-1]

    if file_name is None:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = f"dynamic_wells_ensemble_{current_time}.pkl"

    # Pickle the dictionary to the file
    with open(os.path.join(ensemble_dir, file_name), 'wb') as file:
        pickle.dump(wells_data_ensemble, file)

def save_grids_dynamic_output(ensemble_dir, grids_data_ensemble, file_name = None):
    """
    Saves the grids data ensemble to a pickle file in the ensemble directory.

    Parameters
    ----------
    ensemble_dir : str
        The location of the ensemble directory.
    grids_data_ensemble : list
        A list of dictionaries, where each dictionary contains the data for a single grid
    file_name : str, optional
        The name of the file to save the data to. If not provided, a default name will be used.

    Returns
    -------
    None
    """
    
    
    iter_dir = glob.glob(ensemble_dir + '/iter*')
    if iter_dir:
        ensemble_dir = iter_dir[-1]

    if file_name is None:
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = f"dynamic_grid_ensemble_{current_time}.pkl"

    # Pickle the dictionary to the file
    with open(os.path.join(ensemble_dir, file_name), 'wb') as file:
        pickle.dump(grids_data_ensemble, file)

## clean ensemble run workflow
def clean_ensemble_folder(ensemble_dir, delete_sr3 = False, verbose = False):
    ensemble_dir = _check_iterative_n_replace(ensemble_dir)
    print(ensemble_dir)
    """
    Deletes all the files in the ensemble directory that are not needed for rerunning the ensemble.

    Parameters
    ----------
    ensemble_dir : str
        The location of the ensemble directory.
    delete_sr3 : bool, optional
        If True, deletes the SR3 files in the ensemble directory. Defaults to False.

    Returns
    -------
    None
    """
    files_to_remove = []
    for extension in ['inc','out', 'rst']:
        files_to_remove.extend(glob.glob(ensemble_dir + f'/real_*/*.{extension}'))
    if delete_sr3:
        files_to_remove.extend(glob.glob(ensemble_dir + f'/real_*/*.sr3'))
    if verbose:
        print('files_to_remove', files_to_remove)
    for file in files_to_remove:
        os.remove(file)
