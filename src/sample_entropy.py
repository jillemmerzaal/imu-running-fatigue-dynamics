import numpy as np
import math
import os
import json

from src.engine import engine
from src.fileparts import fileparts
from src.zsave import zsave
from src.fldOrganize import fld_organize


# some hard coded variables
# Yentes, J. M., Hunt, N., Schmid, K. K., Kaipust, J. P., McGrath, D., & Stergiou, N. (2013).
# The appropriate use of approximate entropy and sample entropy with short data sets.
# Annals of biomedical engineering, 41, 349-365.
TOL = 0.2
DIM = 2
CHNS = ["lumbar_aresvec", "lumbar_avert", "lumbar_amedlat", "lumbar_aantpost"]

def sampen_analysis(fld, sfld):
    fl = engine(fld=os.path.join(fld, sfld), extension=".json")
    for f in fl:

        directory, filename, extension = fileparts(f)
        print("Entropy analysis for {}".format(filename))

        with open(f, 'r') as fn:
            data = json.load(fn)

        # loop over channels
        for c in CHNS:
            sampen = sample_entropy(data[c]["line"])
            data[c]["event"].update({"SampleEntropy": [sampen, 0, 0]})

        # save everything to file
        fname = f"{filename}.json"
        zsave(os.path.join(fld, fname), data)
        fld_organize(fld, "nld", fname, False)


def sample_entropy(ndata, **kwargs):
    """
    This function calculates the sample entropy of a given signal. The tolerance is set at 0.2 with a dimension of 2.

    :param ndata: dictionary containing all data.
    :param kwargs: "event" provides the last step index. Used as input for the Euclidean
    norm. If left empty, the entire timeseries will be analysed.
    :return: sampen; a single float of the calculated sample entropy.
    """

    # norm = euclidean_norm(data, keys=ch)

    if kwargs:
        event = kwargs.get("event")
        ndata = ndata[:event]

    N = len(ndata)
    r = TOL * np.std(ndata)
    m = DIM

    # calculate B_i
    matches = np.zeros((m, N)) * np.nan
    for i in range(m):
        matches[i, :N - i] = ndata[i:]

    matches = matches.T
    matches_total = np.zeros((matches.shape[0] - 1))

    for i in range(matches.shape[0] - 1):
        lower_bounds = matches[i, :] - r
        upper_bounds = matches[i, :] + r

        match_is_in_range = np.sum((matches >= lower_bounds) & (matches <= upper_bounds), axis=1)
        matches_total[i] = np.sum(match_is_in_range == m) - 1

    B_i = (np.sum(matches_total) / (N - m)) / (N - m)

    del matches, matches_total

    # calculate A_i
    matches = np.zeros((m + 1, N)) * np.nan
    for i in range(m + 1):
        matches[i, :N - i] = ndata[i:]

    matches = matches.T
    matches_total = np.zeros((matches.shape[0] - 1))

    for i in range(matches.shape[0] - 2):
        lower_bounds = matches[i, :] - r
        upper_bounds = matches[i, :] + r

        match_is_in_range = np.sum((matches >= lower_bounds) & (matches <= upper_bounds), axis=1)
        matches_total[i] = np.sum(match_is_in_range == m + 1) - 1

    A_i = (np.sum(matches_total) / (N - (m + 1))) / (N - m)

    # Calculate sample entropy
    A = (((N - m - 1) * (N - m)) / 2) * A_i
    B = (((N - m - 1) * (N - m)) / 2) * B_i

    sampen = -math.log(A / B)

    return sampen
