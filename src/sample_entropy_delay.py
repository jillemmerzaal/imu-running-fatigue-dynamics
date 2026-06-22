import numpy as np
import math
import os
import json

from src.engine import engine
from src.fileparts import fileparts
from src.zsave import zsave
from src.fldOrganize import fld_organize

import EntropyHub as EH
import statsmodels.tsa.stattools as stattools

DIM = 2
CHNS = ["lumbar_aresvec", "lumbar_avert", "lumbar_amedlat", "lumbar_aantpost"]


def sampen_analysis_delay(fld, sfld):
    fl = engine(fld=os.path.join(fld, sfld), extension=".json")
    for f in fl:

        directory, filename, extension = fileparts(f)
        print("Entropy analysis for {}".format(filename))

        with open(f, 'r') as fn:
            data = json.load(fn)

        # loop over channels
        for c in CHNS:
            sampen = sampen_delay(data[c]["line"])
            data[c]["event"].update({"SampleEntropyDelay": [sampen, 0, 0]})

        # save everything to file
        fname = f"{filename}.json"
        zsave(os.path.join(fld, fname), data)
        fld_organize(fld, "nld", fname, False)

def sampen_delay(data):

    _lags = len(data)
    autocorr = stattools.acf(data, nlags=_lags)
    drop_value = 1 / np.exp(1)
    TAU = np.argmax(autocorr < drop_value)
    Samp, A, B = EH.SampEn(Sig=data, m=2, tau=int(TAU), r=None)

    return Samp[-1]
