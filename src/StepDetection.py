from scipy import signal
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def step_dectection_analysis(data,  vt, ap, ml, fs ):
    accls = pd.DataFrame()
    accls["ML"] = data[ml]["line"]
    accls["AP"] = data[ap]["line"]
    accls["VT"] = data[vt]["line"]

    # filter acceleration data with a 4th order, zero-phase Butterworth filter at 10 Hz
    fc = 10  # Cut-off frequency of the filter
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(N=4, Wn=w, btype='low') # filter parameters

    accls_filter = pd.DataFrame()
    for ch in accls.columns:
        accls_filter[ch] = signal.filtfilt(b, a, accls[ch])

    IC, TO = step_detection_back(data=accls_filter, sample_rate=fs)

    return IC, TO

def step_detection_back(data, sample_rate):
    """
    step detection. Benson 2019 method which is a modification previously described accelerometer patterns by Strohrmann et al. [14] and Lee et al. [19]
    This step detection function only needs the acceleration signal of the lower back sensor.

    step 1: find major pos_peaks in vertical acceleration signal, with at least 0.25s in between pos_peaks

    step 2: for each VT peak
        a: between previous and current VT peak, find negative pos_peaks in the AP acceleration
        b: set the minimum as IC
        c: set the previous negative peak as TO

    step 3: some contingencies when 0, 1 or more than 2 peaks in the AP acceleration signal are found.

    :param data: pandas DataFrame of the filtered acceleration signal with column names ["VT", "AP", "ML"]
    :param sample_rate: sampling rate of the data in Hz
    :return TO: List of all indicis of the Toe offs
    :return IC: List of all indicis of the initial contacts
    """
    # select major positive peaks in the vertical acceleration with at least 0.25 second distance
    pos_peaks, peak_properties = signal.find_peaks(data["VT"], distance=sample_rate * 0.25)

    # plot find peaks results
    fig = make_subplots(rows=2, cols=1,)
    fig.add_trace(go.Scatter(x=np.arange(0,len(data["VT"])), y=data["VT"], name="filtered VT acc"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=pos_peaks, y=data["VT"][pos_peaks], mode="markers", name="max in VT acceleration"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(y=data["VT"], name="filtered VT acc"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=pos_peaks, y=data["VT"][pos_peaks], mode="markers",
                             name="max in VT acceleration"),
                  row=2, col=1)

    # find the negative peaks in AP signal in between two positive peaks TEST2
    TO=[]
    IC=[]
    for i in range(1, len(pos_peaks)):
        # Find negative peaks in AP (Y) signal between two VT peaks
        neg_peaks, properties = signal.find_peaks(-data["AP"][pos_peaks[i - 1]:pos_peaks[i]])
        neg_indices = neg_peaks + pos_peaks[i - 1]


        if len(neg_indices) ==2: # easiest option. If two negative peaks are found
            IC.append(neg_indices[1])
            TO.append(neg_indices[0])

        elif len(neg_indices) == 0: # If no peaks are found
            # Find minimum of AP data in the window
            min_idx = np.argmin(data["AP"][pos_peaks[i - 1]:pos_peaks[i]]) + pos_peaks[i - 1]
            IC.append(min_idx)

            # find maximum of AP data in the window
            max_idx = np.argmax(data["AP"][pos_peaks[i - 1]:pos_peaks[i]]) + pos_peaks[i - 1]
            # find slope between max of AP and IC
            slope = np.diff(data["AP"][max_idx:min_idx]) * sample_rate

            if len(slope) < 3:
                TO.append(max_idx)
            else:
                slope_peaks, _ = signal.find_peaks(slope)
                if len(slope_peaks) == 0:
                    TO.append(max_idx)
                else:
                    max_slope_idx = slope_peaks[np.argmax(slope[slope_peaks])] + max_idx
                    TO.append(max_slope_idx)

        elif len(neg_indices) == 1: # If only one negative peak is found
            # increase the range at the end by about 3 frames, and see if this changes the number peaks
            neg_peaks_2, _ = signal.find_peaks(-data["AP"][pos_peaks[i - 1]:pos_peaks[i]+3])
            neg_indices_2 = neg_peaks_2 + pos_peaks[i - 1]

            # if the length of the peaks has increased.
            if len(neg_indices_2) > len(neg_indices):
                if len(neg_indices_2) == 2:
                    IC.append(neg_indices_2[1])
                    TO.append(neg_indices_2[0])
                elif len(neg_indices_2) > 2:
                    neg_indices_2 = [idx for idx in neg_indices_2 if idx < pos_peaks[i]] # remove all indices beyond the positive peak index.
                    if len(neg_indices_2) == 2:
                        IC.append(neg_indices_2[1])
                        TO.append(neg_indices_2[0])
                    else:
                        pk_order = np.argsort(data["AP"][neg_indices])  # Descending order of peak heights
                        idx_order = np.argsort(-neg_indices)  # Descending order of peak positions
                        ref_rank = np.arange(1, len(neg_indices) + 1)
                        pk_rank = np.zeros_like(ref_rank)
                        idx_rank = np.zeros_like(ref_rank)
                        for p in range(len(neg_indices)):
                            pk_rank[p] = ref_rank[np.where(pk_order == p)[0][0]]
                            idx_rank[p] = ref_rank[np.where(idx_order == p)[0][0]]

                        mean_rank = np.mean(np.column_stack((pk_rank, idx_rank)), axis=1)
                        low_rank_idx = np.argmin(mean_rank)
                        IC.append(neg_indices[low_rank_idx])

                        # Remove peaks after the selected IC
                        neg_indices = neg_indices[:low_rank_idx]
                        if len(neg_indices) > 0:
                            # Remove peaks too close to the previous IC
                            neg_indices = [idx for idx in neg_indices if abs(idx - IC[-2]) / sample_rate >= 0.1]

                        if len(neg_indices) == 0:
                            max_idx = np.argmax(data['AP'][pos_peaks[i - 1]:IC[-1]]) + pos_peaks[i - 1]
                            slope = np.diff(data["AP"][max_idx:IC[-1]]) * sample_rate
                            if len(slope) < 3:
                                TO.append(max_idx)
                            else:
                                slope_peaks, _ = signal.find_peaks(slope)
                                if len(slope_peaks) == 0:
                                    TO.append(max_idx)
                                else:
                                    max_slope_idx = slope_peaks[np.argmax(slope[slope_peaks])] + max_idx
                                    TO.append(max_slope_idx)
                        else:
                            TO.append(max(neg_indices))


            else:
                IC.append(neg_indices[0])

                # find maximum of AP data in the window
                max_idx = np.argmax(data["AP"][pos_peaks[i - 1]:neg_indices[0]]) + pos_peaks[i - 1]
                # Compute slope and find the max slope
                slope = np.diff(data[max_idx:IC[-1], 1]) * sample_rate
                if len(slope) < 3:
                    TO.append(max_idx)
                else:
                    slope_peaks, _ = signal.find_peaks(slope)
                    if len(slope_peaks) == 0:
                        TO.append(max_idx)
                    else:
                        max_slope_idx = slope_peaks[np.argmax(slope[slope_peaks])] + max_idx
                        TO.append(max_slope_idx)

        else: # If more than two peaks are found
            pk_order = np.argsort(data["AP"][neg_indices])  # Descending order of peak heights
            idx_order = np.argsort(-neg_indices)  # Descending order of peak positions
            ref_rank = np.arange(1, len(neg_indices) + 1)
            pk_rank = np.zeros_like(ref_rank)
            idx_rank = np.zeros_like(ref_rank)
            for p in range(len(neg_indices)):
                pk_rank[p] = ref_rank[np.where(pk_order == p)[0][0]]
                idx_rank[p] = ref_rank[np.where(idx_order == p)[0][0]]

            mean_rank = np.mean(np.column_stack((pk_rank, idx_rank)), axis=1)
            low_rank_idx = np.argmin(mean_rank)
            IC.append(neg_indices[low_rank_idx])

            # Remove peaks after the selected IC
            neg_indices = neg_indices[:low_rank_idx]
            if len(neg_indices) > 0:
                # Remove peaks too close to the previous IC
                neg_indices = [idx for idx in neg_indices if abs(idx - IC[-2]) / sample_rate >= 0.1]

            if len(neg_indices) == 0:
                max_idx = np.argmax(data['AP'][pos_peaks[i - 1]:IC[-1]]) + pos_peaks[i - 1]
                slope = np.diff(data["AP"][max_idx:IC[-1]]) * sample_rate
                if len(slope) < 3:
                    TO.append(max_idx)
                else:
                    slope_peaks, _ = signal.find_peaks(slope)
                    if len(slope_peaks) == 0:
                        TO.append(max_idx)
                    else:
                        max_slope_idx = slope_peaks[np.argmax(slope[slope_peaks])] + max_idx
                        TO.append(max_slope_idx)
            else:
                TO.append(max(neg_indices))

    IC = [val for val in IC if val != 0]
    TO = [val for val in TO if val != 0]

    fig.add_trace(go.Scatter(y=data["AP"], name="filtered AP acc"), row=2, col=1)
    fig.add_trace(go.Scatter(x=IC, y=data["AP"][IC], mode="markers", name="Initial contact (IC)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=TO, y=data["AP"][TO], mode="markers", name="Toe off (TO)"), row=2, col=1)

    for ic in IC:
        fig.add_annotation(x=ic, y=data["AP"][ic],
                           text="IC",
                           showarrow=True,
                           arrowhead=1,
                           row=2, col=1)

    for to in TO:
        fig.add_annotation(x=to, y=data["AP"][to],
                           text="TO",
                           showarrow=True,
                           arrowhead=1,
                           row=2, col=1
                           )

    fig.show()

    return IC, TO