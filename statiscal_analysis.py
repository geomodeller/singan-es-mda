import numpy as np
from skimage.metrics import structural_similarity as ssim
from esmda_utils import generate_sample_from_xvar



def calculate_ssim_from_reals(ground_truth: np.ndarray,
                                ensemble_reals:np.ndarray,
                                iter = None) -> np.ndarray:
    if iter is not None:
        ensemble_reals = ensemble_reals[iter]

    if ground_truth.shape == ensemble_reals.shape:
        return ssim(ground_truth, ensemble_reals, data_range=ensemble_reals.max() - ensemble_reals.min())
    elif (ground_truth.shape != ensemble_reals.shape) and (ensemble_reals.shape[1:] == ground_truth.shape):
        return np.array([ssim(ground_truth, sample, data_range=sample.max() - sample.min()) for sample in ensemble_reals])
    else:
        assert False, "Input shapes are not compatible."

def calculate_ssim(ground_truth: np.ndarray,
                   ensemble:np.ndarray,
                   iter = None) -> np.ndarray:
    if iter is not None:
        ensemble = ensemble[iter]
    if len(ensemble.shape) == 2:   
        ensemble = generate_sample_from_xvar(ensemble)

    if ground_truth.shape == ensemble.shape:
        return ssim(ground_truth, ensemble, data_range=ensemble.max() - ensemble.min())
    elif (ground_truth.shape != ensemble.shape) and (ensemble.shape[1:] == ground_truth.shape):
        return np.array([ssim(ground_truth, sample, data_range=sample.max() - sample.min()) for sample in ensemble])
    else:
        assert False, "Input shapes are not compatible."
def compute_etype_map_of_ensemble(ensemble):
    return np.mean(ensemble, axis=0)

def compute_cross_entorpy_of_ensemble(ensemble, eps=1e-12, verbose = False):
    # Step 1: compute probability distribution of each facies at each voxel
    n_facies = len(np.unique(ensemble))
    probs = np.zeros((n_facies,) + ensemble.shape[1:])

    for facies_id in range(n_facies):
        probs[facies_id] = np.mean(ensemble == facies_id, axis=0)  # average over realizations

    # Step 2: compute entropy (cross-entropy with itself)
    # add a small epsilon to avoid log(0)
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=0)
    if verbose:
        print("Entropy shape:", entropy.shape)  # (16, 32, 32)
        print("Entropy range:", entropy.min(), entropy.max())
    return entropy

def compute_cross_entorpy_btw_two_ensemble(ensemble_a, ensemble_b, eps = 1e-12):
    n_facies = len(np.unique(ensemble_a))
    # Probability from A
    p = np.zeros((n_facies,)+ ensemble_a.shape[1:])
    for c in range(n_facies):
        p[c] = np.mean(ensemble_a == c, axis=0)

    # Probability from B
    q = np.zeros_like(p)
    for c in range(n_facies):
        q[c] = np.mean(ensemble_b == c, axis=0)
        
    # Cross-entropy H(p, q)
    cross_entropy = -np.sum(p * np.log(q + eps), axis=0)
    return cross_entropy