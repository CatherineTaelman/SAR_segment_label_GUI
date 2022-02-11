"""
Markov random field contextual smoothing
"""

from scipy import ndimage
from scipy import optimize
import numpy as np

# MRF smoothing kernel
kernel = np.ones((3, 3, 1)) / 8.0
kernel[1][1][0] = 0.0


def energy_function(data, posteriors):
    """
    data = data as an image
    posteriors = posterior for each class as an image [rows x cols x classes]
    """
    rows, cols, bands = data.shape
    classes = posteriors.shape[-1]
    image_posteriors = posteriors.reshape((rows, cols, classes))
    class_membership_means = np.reshape(
        ndimage.convolve(
            image_posteriors,
            kernel,
            mode='reflect'),
        (rows, cols, classes),
    )

    x = data
    m = class_membership_means
    local_priors = np.exp(x*m)
    local_priors /= np.sum(local_priors, axis=???)
    return np.sum(posteriors * np.log(local_priors))

def mrf_smoothing():

    beta = optimize.minimize_scalar(
        energy_function,
        bounds=(0.1, 50.0),
        method='bounded',
    ).x
