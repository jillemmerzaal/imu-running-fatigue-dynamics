import os
import numpy as np
import json
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ptitprince as pt
from scipy.fft import fft, fftfreq

from src.extract2zoo import extract
from src.engine import engine
from src.fileparts import fileparts
from src.setZoosystem import setZoosystem
from src.add_channel import addchannel_data
from src.grab import grab
from src.TiltCorrection import tilt_algorithm
from src.zsave import zsave
from src.fldOrganize import fld_organize
from src.TiltCorrection  import tilt_correction
from src.symmetry import symmetry_analysis
from src.sample_entropy import sampen_analysis
from src.sample_entropy_delay import sampen_analysis_delay
from src.ldlj import ldlj_analysis
from src.LyE import lye_analysis
from src.RMS import rms_analysis

## hardcode some variables
# CONS = ["Pre", "Post"]
# SENSORS = ["lumbar"]
SAMPLE_RATE = 128

## set relevant directories
fld_root = os.getcwd()
fld = os.path.join(fld_root, 'data40')

#%%
fld_mat = os.path.join(fld, 'mat2zoo')
if os.path.exists(fld_mat):
    print('Removing old mat2zoo folder...')
    shutil.rmtree(fld_mat)

#
extract(fld)
#
print("_" * 100)

# %% tilt correction
fld_tilt = os.path.join(fld, 'tiltcorrected')
if os.path.exists(fld_tilt):
    print('Removing old tiltcorrected folder...')
    shutil.rmtree(fld_tilt)

tilt_correction(fld=fld, sfld="mat2zoo")

#%% Formal analysis
fld_nld = os.path.join(fld, "nld")
if os.path.exists(fld_nld):
    print('Removing old nld folder...')
    shutil.rmtree(fld_nld)

sfld = "tiltcorrected"

# Linear metrics
# step symmetry and stride regularity
symmetry_analysis(fld=fld, sfld=sfld)

sfld = "nld"
# root-mean-square --> mesure of variability,
rms_analysis(fld=fld, sfld=sfld)
# impact loading?

# non-linear dynamics
# sample entropy --> complexity
sampen_analysis(fld=fld, sfld=sfld)

# sample entropy with delay
sampen_analysis_delay(fld=fld, sfld=sfld)

# Log dimensionless jerk --> smoothness
ldlj_analysis(fld=fld, sfld=sfld)

# Lyapunov exponent --> dynamic stability
lye_analysis(fld=fld, sfld=sfld)

#%% fft
fl = engine(fld=os.path.join(fld, sfld), extension=".json")
CHNS = ["lumbar_aresvec", "lumbar_avert", "lumbar_amedlat", "lumbar_aantpost"]

for f in fl:
    directory, filename, extension = fileparts(f)
    with open(f, 'r') as fn:
        data = json.load(fn)

    N = len(data["lumbar_avert"]["line"])
    T = 1/SAMPLE_RATE

    for c in CHNS:
        direction = c.split("_")[1]
        yf = fft(data[c]["line"])
        yf = abs(yf)
        xf = fftfreq(N, T)[:N//2]
        addchannel_data(data, f"yf_{direction}", yf, "Video")
        addchannel_data(data, f"xf_{direction}", xf, "Video")

    fname = f"{filename}.json"
    zsave(os.path.join(fld, fname), data)

    fld_organize(fld, "nld", fname, False)

#%% test
# Results table
sfld = "nld"
fld_stats = os.path.join(fld, "stats")
if os.path.exists(fld_stats):
    print('Removing old stats folder...')
    shutil.rmtree(fld_stats)

os.makedirs(fld_stats)

fl = engine(fld=os.path.join(fld, sfld), extension=".json")
events = ["SampleEntropy", "SampleEntropyDelay", "LDLJ", "StepSymmetry", "StrideSymmetry", "LyE_s", "LyE_l", "RMS", "RMSratio"]
for e in events:
    results_table = pd.DataFrame(columns=["subj_id", "condition", "sex"])
    for f in fl:
        directory, filename, extension = fileparts(f)
        with open(f, "r") as fn:
            data=json.load(fn)

        # get characteristics from filename
        new_row = {'subj_id': filename.split("_")[0], 'condition': filename.split("_")[2], 'sex': filename.split("_")[1],}

        # get event data per channel
        eventval={}
        channels = data["zoosystem"]["Video"]["Channels"]
        for c in channels:
            if e in data[c]["event"]:
                direction = c.split("_")[1]
                eventval.update({f"{direction}": data[c]["event"][e][0]})

        new_row.update(eventval)
        results_table = pd.concat([results_table, pd.DataFrame([new_row])], ignore_index=True)

    # write to excel. Each event has its own sheet
    f_results =  os.path.join(fld_stats, "results.xlsx")
    if os.path.exists(f_results):
        with pd.ExcelWriter(f_results, engine='openpyxl', mode='a') as writer:
            results_table.to_excel(writer, sheet_name=e)
    else:
        with pd.ExcelWriter(f_results, engine='openpyxl') as writer:
            results_table.to_excel(writer, sheet_name=e)
