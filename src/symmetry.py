import scipy
import json
import os
import statsmodels.tsa.stattools as stattools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from src.engine import engine
from src.fileparts import fileparts
from src.add_channel import addchannel_data
from src.zsave import zsave
from src.fldOrganize import fld_organize


# some hard coded variables
# deductively chosen to extract only the relevant and dominant pos_peaks.
MIN_DISTANCE = 30
MIN_HEIGHT = 0.10
CHNS = ["lumbar_avert", "lumbar_amedlat", "lumbar_aantpost"]
VEC = "lumbar_aresvec"

def symmetry_analysis(fld, sfld):
    fl = engine(fld=os.path.join(fld, sfld), extension=".json")
    for f in fl:
        directory, filename, extension = fileparts(f)

        with open(f, 'r') as fn:
            data = json.load(fn)

        # Find the step frequency based on the auto-correlation of the vector
        vec_data = data[VEC]["line"]
        d_1, ad_1, d_2, ad_2, autocorr_vec = symmetry(vec_data)

        # add the data to the channels
        addchannel_data(data, "autocorr_aresvec", autocorr_vec, "Video")
        data["autocorr_aresvec"]["event"] = {"d1": [int(d_1), 0, 0],
                                             "StepSymmetry": [float(ad_1), 0, 0],
                                             "d2": [int(d_2), 0, 0],
                                             "StrideSymmetry": [float(ad_2), 0, 0]}

        # loop over the channels
        # pre-determine figure
        fig = make_subplots(rows=2, cols=3,)
        colors = ["#284377", "#CC6600", "#1e4d2b"]
        # find the relevant pos_peaks based on the step en stride frequency previously determined.
        for col, c in enumerate(CHNS):
            column = col + 1
            ndata = data[c]["line"]
            _, _, _, _, autocorr =symmetry(ndata)

            if "medlat" in c:
                ad1 = autocorr[1:d_1+10].min()
            else:
                ad1 = autocorr[1:d_1+10].max()

            ad2 = autocorr[d_1:d_2+10].max()

            #add the directional data to the channels
            direction = c.split("_")[1]
            addchannel_data(data, f"autocorr_{direction}", autocorr, "Video")
            data[f"autocorr_{direction}"]["event"] = {"StepSymmetry": [float(ad1), 0, 0],
                                                      "StrideSymmetry": [float(ad2), 0, 0]}

            # plot the results
            fig.add_trace(go.Scatter(x=np.arange(len(autocorr)), y=autocorr, name="Autocorrelation", marker=dict(color=colors[col])),
                          row=1, col=column)
            fig.add_trace(go.Scatter(x=[d_1, d_2], y=autocorr[[d_1, d_2]], mode="markers+text", name="Peak Autocorrelation",
                                     text=["d1", "d2"], textposition="bottom left", marker=dict(color=colors[col], size=8)),
                          row=1, col=column)

            fig.add_trace(go.Scatter(x=np.arange(len(ndata)), y=ndata, name=CHNS[0], marker=dict(color=colors[col])),
                          row=2, col=column)

            # update axis
            fig.update_xaxes(range=[0, 250], row=1, col=column)
            fig.update_yaxes(title_text=f"{c} (g)", row=2, col=column)

            # update figure layout
        fig.update_layout(title_text=filename)

        # fig.show()

        # save everything to file
        fname = f"{filename}.json"
        zsave(os.path.join(fld, fname), data)

        nfld = fld_organize(fld, "nld", fname, True)
        figname = f"{filename}.html"
        fig.write_html(os.path.join(nfld, figname))


def symmetry(ndata, **kwargs):
    """
    This function determines the time delay and level of correlation of a given signal using the autocorrelation coefficient

    :param ndata: an array containing the data.
    :param kwargs: "event" provides the last step index. Used as input for the Euclidean
    norm. If left empty, the entire timeseries will be analysed.
    :return d_1: the time delay of the first dominant peak of the autocorrelation signal.
    :return ad_1: the strength of the correlation of the first dominant peak.
    :return d_2: time delay of the second dominant peak of the autocorrelation signal
    :return ad_2: strength of the correlation of the second dominant peak
    :return autocorr: the autocorrelation signal from the zero phase to signal length.
    """

    if kwargs:
        event = kwargs.get("event")
        ndata = ndata[:event]

    if kwargs.get("medlat"):
        medlat = kwargs.get("medlat")

    _lags = len(ndata)
    autocorr = stattools.acf(ndata, nlags=_lags)
    peaks_auto, peak_properties = scipy.signal.find_peaks(autocorr, distance=MIN_DISTANCE, height=MIN_HEIGHT)

    try:
        d_1 = peaks_auto[0]
        ad_1 = peak_properties["peak_heights"][0]
        d_2 = peaks_auto[1]
        ad_2 = peak_properties["peak_heights"][1]

        if d_1 < 15:
            d_1 = peaks_auto[1]
            ad_1 = peak_properties["peak_heights"][1]
            d_2 = peaks_auto[2]
            ad_2 = peak_properties["peak_heights"][2]

    except IndexError:
        d_1 = np.nan
        ad_1 = np.nan
        d_2 = np.nan
        ad_2 = np.nan


    return d_1, ad_1, d_2, ad_2, autocorr
