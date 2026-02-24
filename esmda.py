import numpy as np
def ES(OBS, static, dynamic, alpha=1, stdErrOfDynamic_percentage=0.05, add_noise = False):
    """
    Ensemble Smoother function that assimilates observations into an ensemble.

    Parameters:
    OBS (numpy.ndarray): Array of observations.
    static (numpy.ndarray): Array of static variables. should be provided (number of ensemble x number of static variables).
    dynamic (numpy.ndarray): Array of dynamic variables. should be provided (number of ensemble x number of dynamic variables).
    alpha (float): Smoothing factor (default is 1).
    stdErrOfDynamic (float): Standard error of dynamic variables (default is 0.1).
    add_noise (bool): Add noise to dynamic variables (default is False).

    Returns:
    numpy.ndarray: Array of updated static variables.
    """

    # Reshape static and dynamic arrays
    static = static.reshape(static.shape[0], -1).T
    dynamic = dynamic.reshape(dynamic.shape[0], -1).T
    stdErrOfDynamic = stdErrOfDynamic_percentage * dynamic.std(axis=1)
    if add_noise:
        dynamic = dynamic + np.random.normal(loc = 0, scale = stdErrOfDynamic.reshape(-1,1), size = dynamic.shape)

    # Concatenate static and dynamic arrays
    ensemble = np.concatenate((static, dynamic), axis=0)
    No_realization = ensemble.shape[1]

    # Calculate ensemble mean
    En_Mean = ensemble.mean(axis=1).reshape(-1, 1)
    En_Mean = np.repeat(En_Mean, No_realization, axis=1)

    # Reshape OBS array
    OBS = OBS.reshape(-1, 1)
    num_static = static.shape[0]
    num_dynamic = dynamic.shape[0]
    num_state_vector = num_dynamic + num_static
    ref_OBS = np.repeat(OBS, No_realization, axis=1)


    ## This is where we apply EnKF/ES:
    # Create Cd matrix
    sizeOfCd = num_dynamic
    Cd = np.zeros((sizeOfCd, sizeOfCd))
    for i in range(sizeOfCd):
        Cd[i, i] = stdErrOfDynamic[i] ** 2

    # Create H matrix
    H = np.zeros((num_dynamic, num_state_vector))
    for i in range(num_dynamic):
        H[i, num_static + i] = 1

    # Calculate Le and H_Le
    Le = 1 / np.sqrt(num_state_vector - 1) * (ensemble - En_Mean)
    H_Le = np.dot(H, Le)

    # Calculate Kalman Gain, K
    K_ = np.dot(np.dot(Le, H_Le.T), np.linalg.inv(np.dot(H_Le, H_Le.T) + Cd * alpha))

    # Add stdErroOfDynamic to refOBS
    for i in range(num_dynamic):
        for j in range(No_realization):
            ref_OBS[i, j] = ref_OBS[i, j] + np.random.normal(scale=stdErrOfDynamic[i] * (alpha) ** (1/2))

    # Assimilate En
    ensemble_new = ensemble + np.dot(K_, ref_OBS - np.dot(H, ensemble))
    static_new = ensemble_new[:num_static, :]

    return static_new.T 


def ensemble_smoother(obs, static, dynamic, alpha=1.0, stderr_percentage=0.05, add_noise=False):
    """
    [CHAT-GPT assisted code]
    Ensemble Smoother for data assimilation of static and dynamic variables.

    Parameters:
    ----------
    obs : np.ndarray
        Observation vector (n_dynamic x 1 or flattened).
    static : np.ndarray
        Static ensemble matrix (n_ensemble x n_static).
    dynamic : np.ndarray
        Dynamic ensemble matrix (n_ensemble x n_dynamic).
    alpha : float, optional
        Inflation (smoothing) factor. Default is 1.0.
    stderr_percentage : float, optional
        Standard error as a percentage of dynamic std. Default is 0.05 (5%).
    add_noise : bool, optional
        Whether to add noise to dynamic ensemble before assimilation.

    Returns:
    -------
    np.ndarray
        Updated static ensemble (n_ensemble x n_static).
    """

    # Transpose for consistent shape: (n_vars x n_ensemble)
    static = static.T  # shape: (n_static x n_ensemble)
    dynamic = dynamic.T  # shape: (n_dynamic x n_ensemble)
    n_ensemble = static.shape[1]
    n_static = static.shape[0]
    n_dynamic = dynamic.shape[0]

    # Compute dynamic standard error
    dynamic_std = dynamic.std(axis=1, ddof=1)
    stderr_dynamic = stderr_percentage * dynamic_std

    # Optional: Add noise to dynamic
    if add_noise:
        noise = np.random.normal(loc=0.0, scale=stderr_dynamic[:, None], size=dynamic.shape)
        dynamic += noise

    # Combine static and dynamic into full state ensemble
    state = np.vstack((static, dynamic))  # shape: (n_state x n_ensemble)
    n_state = state.shape[0]

    # Compute ensemble mean and anomalies
    state_mean = np.mean(state, axis=1, keepdims=True)
    state_anomalies = (state - state_mean) / np.sqrt(n_ensemble - 1)

    # Observation operator H (selects only dynamic part)
    H = np.zeros((n_dynamic, n_state))
    H[np.arange(n_dynamic), n_static + np.arange(n_dynamic)] = 1

    # Observation perturbation
    obs = obs.reshape(-1, 1)
    perturbed_obs = obs + np.random.normal(scale=(stderr_dynamic * np.sqrt(alpha))[:, None], size=(n_dynamic, n_ensemble))

    # Observation anomalies
    H_state_anomalies = H @ state_anomalies

    # Observation error covariance matrix Cd
    Cd = np.diag((stderr_dynamic ** 2))

    # Kalman Gain
    cov_ensemble_obs = state_anomalies @ H_state_anomalies.T
    cov_obs_obs = H_state_anomalies @ H_state_anomalies.T + alpha * Cd
    kalman_gain = cov_ensemble_obs @ np.linalg.inv(cov_obs_obs)

    # Update state with observations
    innovations = perturbed_obs - H @ state
    updated_state = state + kalman_gain @ innovations

    # Extract updated static part and transpose back to (n_ensemble x n_static)
    updated_static = updated_state[:n_static, :].T

    return updated_static
