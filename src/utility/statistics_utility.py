"""
    Calculating the mean and confidence interval for sampled data
"""

import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    """
        Taken from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    :param data:
    :param confidence:
    :return:
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h
