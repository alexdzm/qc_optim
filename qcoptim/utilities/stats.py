"""
"""

import numpy as np


def bootstrap_resample(stat_func, empirical_distribution, num_bootstraps,
                       return_dist=False):
    """
    Calculate the boostrap mean and standard-error of `stat_func` applied to
    `empirical_distribution` dataset. Optionally return the list of resampled
    values of the estimator, instead of the standard error.

    Parameters
    ----------
    stat_func : Callable
        Function to evaluate on each bootstrap resample e.g. np.std
    empirical_distribution : np.ndarray
        Data to resample
    num_bootstraps : int
        Number of bootstrap resamples to perform
    return_dist : boolean, default False
        If True, return the list of resampled values of the estimator instead
        of the standard error (see Returns)

    Returns
    -------
    mean_estimate : float
    standard_error OR resampled_values : float OR list[float]
        If `return_dist=True` returns list of resampled values, else returns
        standard error
    """
    num_data = empirical_distribution.size

    try:
        # try to vectorise the bootstrapping, unless the size of the resulting
        # array will be too large (134217728 is approx the size of a 1GB
        # float64 array)
        if num_bootstraps * num_data > 134217728:
            raise TypeError

        # vectorisation will also fail if `stat_func` does not have an `axis`
        # kwarg, which will raise a TypeError here
        resample_indexes = np.random.randint(
            0, num_data, size=(num_bootstraps, num_data))
        resampled_estimator = stat_func(
            empirical_distribution[resample_indexes], axis=1
            )
    except TypeError:
        # more memory safe but much slower
        resampled_estimator = np.zeros(num_bootstraps)
        for boot in range(num_bootstraps):
            resample_indexes = np.random.randint(0, num_data, size=num_data)
            resampled_estimator[boot] = stat_func(
                empirical_distribution[resample_indexes])

    if return_dist:
        # bootstrap estimate and list of resampled values
        return (
            np.mean(resampled_estimator),
            resampled_estimator
        )
    # bootstrap estimate and standard deviation
    return (
        np.mean(resampled_estimator),
        np.std(resampled_estimator)
    )
