import os
import glob
import concurrent.futures
from tqdm import tqdm

CMG = r'"C:\Program Files\CMG\GEM\2024.20\Win_x64\EXE\gm202420.exe"'

def delete_no_needed_files(extensions_to_delete = ['rst'], folder = '.'):
    for file in os.listdir(folder):
        if file.endswith(tuple(extensions_to_delete)):
            os.remove(os.path.join(folder, file))

def _check_if_sim_is_done(cmg_data_file, min_required_sim_time = 30):
    if os.path.exists(cmg_data_file.split('.')[0]+'.out') == False:
        return False

    else:
        with open(cmg_data_file.split('.')[0]+'.out', 'r') as f:
            lines = f.readlines()
            f.close()
        sim_time = float(lines[-1].strip().split(' ')[-1])
        if sim_time > min_required_sim_time:
            return True
        else:
            return False
    
def run_cmg_simulator(folder_name,
                      cmg_data_file,
                      CMG = CMG,
                      is_overwrite:bool = False,
                      is_delete_no_needed_file:bool = True,
                      min_required_sim_time:float = 60

                    ):
    """
    Executes a CMG simulation using a specified executable and data file.

    Parameters:
    CMG (str): The path to the CMG executable.
    folder_name (str): The directory where the simulation should be run.
    cmg_data_file (str): The data file used for the CMG simulation.
    """
    current_dir = os.getcwd()
    os.chdir(folder_name)
    if is_overwrite:
        # while True:
        os.system(f'{CMG} -f {cmg_data_file}')
            # if _check_if_sim_is_done(cmg_data_file, 
            #                 min_required_sim_time = min_required_sim_time):
            #     break
    else:
        if os.path.exists(cmg_data_file.split('.')[0]+'.sr3') == False: 
            # while True:
            os.system(f'{CMG} -f {cmg_data_file}')
                # if _check_if_sim_is_done(cmg_data_file, 
                #                          min_required_sim_time = min_required_sim_time):
                #     break

    if is_delete_no_needed_file:
        delete_no_needed_files()
    os.chdir(current_dir)

def run_cmg_for_ensemble(ensemble_dir,
                         cmg_data_file,
                         level_of_direcory = 1,
                         CMG = CMG):
    real_dir = glob.glob(ensemble_dir + '/*'*level_of_direcory)
    for dir in tqdm(real_dir, desc = 'Running CMG ensemble'):
        run_cmg_simulator(dir, cmg_data_file, CMG)

def run_cmg_for_ensemble_parallel(ensemble_dir,
                                  cmg_data_file,
                                  level_of_direcory = 1,
                                  CMG = CMG,
                                  max_workers = 5
                                  ):

    real_dir = glob.glob(ensemble_dir + '/*'*level_of_direcory)
    lst_cmg_data_file = [cmg_data_file for _ in range(len(real_dir))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_cmg_simulator, real_dir, lst_cmg_data_file)

















# ===================================== old code =====================================
# def generate_realization(xvars):
#     """
#     Generates a realization of the given xvars dictionary.

#     Parameters:
#     xvars (dict): A dictionary where the keys are the variable names and the values are lists of two elements that represent the minimum and maximum possible values for that variable.

#     Returns:
#     dict: A dictionary with the same keys as xvars, but with randomly generated values between the minimum and maximum possible values for each variable.
#     """
#     xvars_real = {}

#     for key, config in xvars.items():
#         if isinstance(config, list):  
#             xvars_real[key] = str(round(random.uniform(config[0], config[1]), 6))  
#             continue

#         var_type = config.get("type", "uniform")

#         if var_type == "uniform":
#             min_val, max_val = config.get("range", [0, 1])
#             xvars_real[key] = str(round(random.uniform(min_val, max_val), 6))  

#         elif var_type == "gaussian":
#             mean = config.get("mean", 0)
#             std = config.get("std", 1)
#             min_val, max_val = config.get("range", [-np.inf, np.inf])

#             value = np.random.normal(mean, std)
#             value = max(min(value, max_val), min_val)
#             xvars_real[key] = str(round(value, 6)) 
             
#         elif var_type == "categorical":
#             choices = config.get("choices", ["default"])
#             xvars_real[key] = str(random.choice(choices))

#         else:
#             raise ValueError(f"Unknown variable type: {var_type}")

#     return xvars_real

# def copy_selected_files(file_list, destination_dir, verbose = False):
#     """
#     Copies selected files from a given list to a specified destination directory.

#     Parameters:
#     file_list (list): A list of file paths to be copied.
#     destination_dir (str): The directory where the files will be copied to.
#     verbose (bool, optional): If True, prints additional information about the copy process. Defaults to False.

#     Prints:
#     str: A message indicating whether each file was successfully copied or not found.
#          Also prints a success message if all files are copied successfully when verbose is True.

#     Exceptions:
#     Exception: Prints an error message if any exception occurs during the process.
#     """

#     try:
#         # Ensure destination directory exists
#         if not os.path.exists(destination_dir):
#             os.makedirs(destination_dir)

#         # Loop through each file in the list
#         for file_path in file_list:
#             if os.path.isfile(file_path):
#                 filename = os.path.basename(file_path)
#                 destination_file = os.path.join(destination_dir, filename.split('\\')[-1])

#                 # Copy file to destination
#                 shutil.copy2(file_path, destination_file)
#                 if verbose:
#                     print(f"Copied: {filename}")
#             else:
#                 print(f"File not found: {file_path}")
#         if verbose:
#             print("Selected files copied successfully!")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")

# def replace_words_in_file(input_file, output_file, xvar_real, verbose = False):
#     """
#     Replaces words in a file based on a given dictionary.

#     Parameters:
#     input_file (str): The path to the file to be processed.
#     output_file (str): The path to the output file where the processed content will be written.
#     xvar_real (dict): A dictionary where the keys are the words to be replaced and the values are the replacement values.
#     verbose (bool, optional): If True, prints a success message. Defaults to False.

#     Prints:
#     str: A message indicating whether the file was processed successfully or not. If verbose is True, prints an additional message if the file is processed successfully.

#     Exceptions:
#     Exception: Prints an error message if any exception occurs during the process.
#     """
#     try:
#         # Read the original content
#         with open(input_file, 'r', encoding='utf-8') as file:
#             content = file.read()

#         # Replace words based on the dictionary
#         for key, value in xvar_real.items():
#             content = content.replace(key, value)

#         # Write the modified content to the output file
#         with open(output_file, 'w', encoding='utf-8') as file:
#             file.write(content)
#         if verbose:
#             print("File processed successfully!")
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
        
# def generate_ensemble_folder(ensemble_dir, cmg_template, xvars, num_of_reals, cmg_file = None, return_ensemble = True):
#     """
#     Generates a folder structure for an ensemble of CMG simulations.

#     Parameters:
#     ensemble_dir (str): The name of the top-level directory for the ensemble.
#     cmg_template (str): The directory where the template CMG files are located.
#     xvars (dict): A dictionary where the keys are the variable names and the values are lists of two elements that represent the minimum and maximum possible values for that variable.
#     num_of_reals (int): The number of realizations to generate.

#     Returns:
#     None
#     """
#     if cmg_file is None:
#         cmg_file = glob.glob(f'{cmg_template}/*.dat')[0]
#     else:
#         cmg_file = os.path.join(cmg_template, cmg_file)
    
#     cmg_file_name = cmg_file.split('\\')[-1]
#     include_files = glob.glob(f'{cmg_template}/*.inc')
#     if os.path.exists(ensemble_dir):
#         assert False, f'Folder {ensemble_dir} already exists' 
#     else:
#         os.mkdir(ensemble_dir)

#     ensemble = []
#     for i in range(num_of_reals):
#         # make real folder
#         os.mkdir(os.path.join(ensemble_dir, f'real_{i}'))
#         # copy and paste cmg files
#         copy_selected_files(include_files, os.path.join(ensemble_dir, f'real_{i}'))
            
#         # replace xvars
#         xvar_real = generate_realization(xvars)
#         ensemble.append(xvar_real)
#         replace_words_in_file(cmg_file, os.path.join(ensemble_dir, f'real_{i}\\{cmg_file_name}'), xvar_real)
#     if return_ensemble:
#         return ensemble

# def _check_n_generate_ensemble_folder(ensemble_dir):
#     if os.path.exists(ensemble_dir):
#         assert False, f'Folder {ensemble_dir} already exists' 
#     else:
#         os.mkdir(ensemble_dir)

# def generate_ensemble_folder_iter(ensemble_dir, 
#                                   cmg_template, 
#                                   xvars, 
#                                   num_of_reals, 
#                                   iteration = None,
#                                   return_ensemble = True):
#     cmg_file = glob.glob(f'{cmg_template}/*.dat')[0]
#     cmg_file_name = cmg_file.split('\\')[-1]
#     include_files = glob.glob(f'{cmg_template}/*.inc')

#     # this is to generate run ensemble folders
#     _check_n_generate_ensemble_folder(ensemble_dir)

#     # handling iteration number
#     if iteration is not None:
#         ensemble_dir = os.path.join(ensemble_dir, f'iter_{iteration:02}')
#     else:
#         iter_dirs = glob.glob(f'{ensemble_dir}/iter_*')
#         if len(iter_dirs) > 0:
#             iteration = int(iter_dirs[-1].split('iter_')[-1]) + 1
#             ensemble_dir = os.path.join(ensemble_dir, f'iter_{iteration:02}')
#         else:
#             ensemble_dir = os.path.join(ensemble_dir, f'iter_00')
#     # this is to generate iteration folders
#     _check_n_generate_ensemble_folder(ensemble_dir)

#     ## populate realizations:
#     ensemble = []
#     for i in range(num_of_reals):
#         # make real folder
#         os.mkdir(os.path.join(ensemble_dir, f'real_{i}'))
#         # copy and paste cmg files
#         copy_selected_files(include_files, os.path.join(ensemble_dir, f'real_{i}'))
            
#         # replace xvars
#         xvar_real = generate_realization(xvars)
#         ensemble.append(xvar_real)
#         replace_words_in_file(cmg_file, os.path.join(ensemble_dir, f'real_{i}\\{cmg_file_name}'), xvar_real)
    
#     if return_ensemble:
#         return ensemble

# ================================= end of the old code ================================
