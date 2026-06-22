import numpy as np
import json
import os

from src.engine import engine
from src.fldOrganize import fld_organize
from src.fileparts import fileparts
from src.zsave import zsave


def rms_analysis(fld, sfld):
    CHNS = ["lumbar_aresvec", "lumbar_avert", "lumbar_amedlat", "lumbar_aantpost"]


    fl = engine(fld=os.path.join(fld, sfld), extension=".json")
    for f in fl:

        directory, filename, extension = fileparts(f)
        print("rms analysis for {}".format(filename))
        with open(f, 'r') as fn:
            data = json.load(fn)
            # loop over channels
        for c in CHNS:
            # direction = c.split("_")[1]
            ndata = data[c]["line"]
            rms = root_mean_square(ndata)
            data[c]["event"].update({"RMS": [float(rms), 0, 0],})


        rmsr_vt, rmsr_ml, rmsr_ap = root_mean_square_ratio(data["lumbar_avert"]["event"]["RMS"][0], data["lumbar_amedlat"]["event"]["RMS"][0], data["lumbar_aantpost"]["event"]["RMS"][0])

        data["lumbar_avert"]["event"].update({"RMSratio": [float(rmsr_vt), 0, 0]})
        data["lumbar_amedlat"]["event"].update({"RMSratio": [float(rmsr_ml), 0, 0]})
        data["lumbar_aantpost"]["event"].update({"RMSratio": [float(rmsr_ap), 0, 0]})

        # save everything to file
        fname = f"{filename}.json"
        zsave(os.path.join(fld, fname), data)
        fld_organize(fld, "nld", fname, False)


def root_mean_square(ndata):
    # type check
    if not isinstance(ndata, np.ndarray):
        ndata = np.array(ndata)

    rms = np.sqrt(np.mean(ndata**2))

    return rms


def root_mean_square_ratio(rms_vt, rms_ml, rms_ap):
    rms_t = np.sqrt([rms_vt**2 + rms_ml**2 + rms_ap**2])

    rmsr_vt = rms_vt/rms_t
    rmsr_ml = rms_ml/rms_t
    rmsr_ap = rms_ap/rms_t

    return rmsr_vt, rmsr_ml, rmsr_ap
