import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.add_channel import addchannel_data
from src.engine import engine
from src.fileparts import fileparts
from src.zsave import zsave
from src.fldOrganize import fld_organize


def tilt_correction(fld, sfld):
    fl = engine(fld=os.path.join(fld, sfld), extension=".json")
    fl.sort()

    for f in fl:
        directory, filename, extension = fileparts(f)

        with open(f, 'r') as fn:
            data = json.load(fn)

        ## tilt correction
        temp_v = -np.array(data["Acc_lumbar_X"]["line"]) / 9.81
        temp_ml = np.array(data["Acc_lumbar_Y"]["line"]) / 9.81
        temp_ap = -np.array(data["Acc_lumbar_Z"]["line"]) / 9.81
        # Tilt correction
        data_corrected, TiltAngle_ml, TiltAngle_ap = tilt_algorithm(avert=temp_v, amedlat=temp_ml, aantpost=temp_ap, plot_or_not=0)


        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=True,
                            subplot_titles=("Vertical Acceleration",
                                            "ML tilt angle corrected with {} degrees".format(round(TiltAngle_ml, 2)),
                                            "AP tilt angle corrected with {} degrees".format(round(TiltAngle_ap, 2))))


        # ========================
        fig.add_trace(go.Scatter(x=np.arange(len(temp_v)), y=temp_v, name="raw", marker=dict(color="#CC6600"),
                                 legendgroup="raw"),
                      row=1,col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(temp_v)), y=data_corrected["avert"], name="corrected", marker=dict(color="#1e4d2b"),
                                 legendgroup="corrected"),
                      row=1,col=1)

        fig.add_trace(go.Scatter(x=np.arange(len(temp_ml)), y=temp_ml, name='raw', marker=dict(color="#CC6600"),
                                 legendgroup="raw", showlegend=False),
                      row=2,col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(temp_ml)), y=data_corrected["amedlat"], name="corrected", marker=dict(color="#1e4d2b"),
                                 legendgroup="corrected", showlegend=False),
                      row=2,col=1)

        fig.add_trace(go.Scatter(x=np.arange(len(temp_ap)), y=temp_ap, name="raw", marker=dict(color="#CC6600"),
                                 legendgroup="raw", showlegend=False),
                      row=3,col=1)
        fig.add_trace(go.Scatter(x=np.arange(len(temp_ap)), y=data_corrected["aantpost"], name="corrected", marker=dict(color="#1e4d2b"),
                                 legendgroup="corrected", showlegend=False),
                      row=3,col=1)

        # add corrected data to the zoo file
        for c in data_corrected.keys():
            ndata = data_corrected[c]
            addchannel_data(data, f"lumbar_{c}", ndata, "Video")

        fname = f"{filename}.json"
        zsave(os.path.join(fld, fname), data)

        nfld = fld_organize(fld, "tiltcorrected", fname, True)
        figname = f"{filename}.html"
        fig.write_html(os.path.join(nfld, figname))



def tilt_algorithm(avert, amedlat, aantpost, plot_or_not):
    """
    TiltAlgorithm - to account for gravity and improper tilt alignment of a tri-axial trunk accelerometer.
    Step 1: Extract raw measured (mean) accelerations
    Step 2: Calculate tilt angles
    Step 3: Calculate horizontal dynamic accelerations vectors
    Step 4: Calculate estimated provisional vertical vector
    Step 5: Calculate vertical dynamic vector
    step 6.1:  Calculate the contribution of static components
    step 6.2 Transpose static component matrices
    step 7: Remove the static components from the templates of pre and post

    :param avert: signal predominantly in vertical direction
    :param amedlat: signal predominantly in medio-lateral direction
    :param aantpost: signal predominantly in anterior-posterior direction
    :param plot_or_not: whether to plot the results
    :return: dictionary of the tilt corrected and gravity subtracted vertical, medio-lateral and anterior-posterior
    acceleration signals
    """
    #

    aresvec = np.linalg.norm(x=[avert, amedlat, aantpost], axis=0)

    a_vt = avert.mean()
    a_ml = amedlat.mean()
    a_ap = aantpost.mean()

    # Anterior tilt
    TiltAngle_ap_rad = np.arcsin(a_ap)
    TiltAngle_ap_deg = math.degrees(TiltAngle_ap_rad)

    # mediolateral tilt
    TiltAngle_ml_rad = np.arcsin(a_ml)
    TiltAngle_ml_deg = math.degrees(TiltAngle_ml_rad)

    # Anterior posterior
    a_AP = (a_ap * np.cos(TiltAngle_ap_rad)) - (a_vt * np.sin(TiltAngle_ap_rad))
    # AMediolateral
    a_ML = (a_ml * np.cos(TiltAngle_ml_rad)) - (a_vt * np.sin(TiltAngle_ml_rad))

    # a_vt_prov = a_ap*Sin(theta_ap) + a_vt*Cos(theta_ap)
    a_vt_prov = (a_ap * np.sin(TiltAngle_ap_rad)) + (a_vt * np.cos(TiltAngle_ap_rad))

    # a_VT = a_ml*sin(theta_ml) + a_vt_prov*cos(theta_ml) - 1
    a_VT = (a_ml * np.sin(TiltAngle_ml_rad)) + (a_vt_prov * np.cos(TiltAngle_ml_rad)) - 1

    a_AP_static = a_ap - a_AP
    a_ML_static = a_ml - a_ML
    a_VT_static = a_vt - a_VT

    a_AP_static = np.transpose(a_AP_static)
    a_ML_static = np.transpose(a_ML_static)
    a_VT_static = np.transpose(a_VT_static)

    amedlat2 = amedlat - a_ML_static
    avert2 = avert - a_VT_static
    aantpost2 = aantpost - a_AP_static
    aresvec2 = aresvec - 1

    data_corrected = {'avert': avert2,
                      'amedlat': amedlat2,
                      'aantpost': aantpost2,
                      'aresvec': aresvec2,}

    if plot_or_not == 1:
        f, ax = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=300)
        sns.despine(offset=10)
        f.tight_layout()
        offset = 0.1
        f.subplots_adjust(left=0.15, top=0.95)

        sns.lineplot(avert, ax=ax[0], label='Raw')
        sns.lineplot(avert2, ax=ax[0], label='tilt corrected')
        ax[0].set_ylabel('vert acc (g)')
        ax[0].set_title('Vertical acceleration corrected with {}'.format(np.round(a_VT_static, 2)))

        sns.lineplot(amedlat, ax=ax[1], label='Raw')
        sns.lineplot(amedlat2, ax=ax[1], label='Tilt corrected')
        ax[1].set_ylabel('ml acc (g)')
        ax[1].set_title('Medio-lateral tilt angle corrected with {} degrees'.format(np.round(TiltAngle_ml_deg, 2)))

        sns.lineplot(aantpost, ax=ax[2], label='Raw')
        sns.lineplot(aantpost2, ax=ax[2], label='Tilt corrected')
        ax[2].set_ylabel('ap acc (g)')
        ax[2].set_title('Anterior-posterior tilt angle corrected with {} degrees'.format(np.round(TiltAngle_ap_deg, 2)))

        plt.show()

    return data_corrected, TiltAngle_ml_deg, TiltAngle_ap_deg
