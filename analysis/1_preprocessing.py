import io
import os

import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import PIL
import pyllusion as ill
import scipy.stats

# Convenience functions ======================================================================
mne.set_log_level(verbose="WARNING")


def qc_physio(df, info, sub, plot_ecg=[]):
    # Save ECG plot
    nk.ecg_plot(df, info)
    fig = plt.gcf()
    fig.suptitle(f"{sub}")

    # Remove legend and resize
    [ax.legend().set_visible(False) for ax in fig.axes]
    fig.set_size_inches(fig.get_size_inches() * 0.3)

    plt.close(fig)  # Do not show the plot in the console
    plot_ecg.append(nk.fig2img(fig))

    return plot_ecg


# Processing =================================================================================


def process_tap(sub, qc_tap_ecg=[]):
    # Path to EEG data
    path_eeg = path + sub + "/eeg/"
    path_beh = path + sub + "/beh/"

    # Open TAP file
    file = [file for file in os.listdir(path_eeg) if "TAP" in file]
    file = path_eeg + [f for f in file if ".vhdr" in f][0]
    tap = mne.io.read_raw_brainvision(file, preload=True, verbose=False)

    # Load behavioral data
    file = [file for file in os.listdir(path_beh) if "TAP" in file]
    file = path_beh + [f for f in file if ".tsv" in f][0]
    tap_beh = pd.read_csv(file, sep="\t")

    if sub in ["sub-57"]:
        # LUX was not facing screen initially, was corrected after first (medium pace) TAP
        tap = tap.crop(tmin=100)
    if sub in ["sub-80"]:
        tap = tap.crop(tmin=220)
    if sub in ["sub-91"]:
        # lux fell off during the first trial of TAP
        tap = tap.crop(tmin=110)
    if sub in ["sub-93"]:
        # LUX was not plugged in for first TAP trial but fixed this before the second trial started
        tap = tap.crop(tmin=100)

    photo = tap["PHOTO"][0][0]  # nk.signal_plot(photo[0:10000])
    if sub in ["sub-76"]:
        photo[photo > 5] = 5  # Odd peak

    # Find events and crop just before (1 second +/-) first and after last
    events = nk.events_find(
        photo,
        threshold_keep="below",
        duration_min=50,
        duration_max=500,
    )
    # In milliseconds (= /2 for 2000 Hz)
    onsets_photo = events["onset"] / (tap.info["sfreq"] / 1000)

    # If long first event, remove it
    if (len(onsets_photo) == 421) & (
        np.diff(onsets_photo)[0] > np.mean(np.diff(onsets_photo)[1:61]) * 3
    ):
        onsets_photo = onsets_photo[1::]

    # Manual fix
    if sub in [
        "sub-01",
        "sub-17",
        "sub-22",
        "sub-26",
        "sub-44",
        "sub-59",
        "sub-64",
        "sub-65",
        "sub-94",
        "sub-99",
        "sub-100",
        "sub-101",
        "sub-104",
        "sub-105",
        "sub-108",
        "sub-112",
        "sub-113",
        "sub-115",
    ]:
        onsets_photo = onsets_photo[1::]
    if sub in ["sub-42"]:
        onsets_photo = onsets_photo[1::]
        onsets_photo2 = np.insert(
            onsets_photo, 319, onsets_photo[0] + tap_beh["Tapping_Times"].values[319]
        )
        onsets_photo2 = np.insert(
            onsets_photo2, 353, onsets_photo[0] + tap_beh["Tapping_Times"].values[353]
        )
        onsets_photo2 = np.insert(
            onsets_photo2, 370, onsets_photo[0] + tap_beh["Tapping_Times"].values[370]
        )
        # plt.vlines(onsets_photo[0:420], 0, 1, color="blue")
        # plt.vlines(onsets_photo2[0:420], 0.5, 1, color="green")
        # plt.vlines(
        #     tap_beh["Tapping_Times"].values[0:420] + onsets_photo[0],
        #     1,
        #     2,
        #     color="red",
        # )
        onsets_photo = onsets_photo2
    if sub in ["sub-98"]:
        onsets_photo = np.insert(
            onsets_photo, 166, onsets_photo[0] + tap_beh["Tapping_Times"].values[166]
        )
    if sub in ["sub-99"]:
        tap_beh = tap_beh.loc[2::]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
    if sub in ["sub-48"]:  # Participant started tapping before the first photo
        tap_beh = tap_beh[tap_beh["Condition"] != "Baseline"]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
        onsets_photo = onsets_photo[9::]
    if sub in ["sub-57"]:
        tap_beh = tap_beh[tap_beh["Condition"] != "Baseline"]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
        onsets_photo = onsets_photo[1::]
    if sub in ["sub-80"]:
        tap_beh = tap_beh[tap_beh["Condition"] != "Baseline"]
        tap_beh = tap_beh[tap_beh["Condition"] != "Slower"]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
        onsets_photo = onsets_photo[75::]
    if sub in ["sub-91"]:
        tap_beh = tap_beh[tap_beh["Condition"] != "Baseline"]
        tap_beh = tap_beh[tap_beh["Condition"] != "Slower"]
        tap_beh = tap_beh[tap_beh["Condition"] != "Faster"]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
        onsets_photo = onsets_photo[112::]
    if sub in ["sub-93"]:
        tap_beh = tap_beh[tap_beh["Condition"] != "Baseline"]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"].values - tap_beh["Tapping_Times"].values[0]
        )
    if sub in ["sub-109"]:
        tap_beh = tap_beh.loc[1::]
        tap_beh["Tapping_Times"] = (
            tap_beh["Tapping_Times"] - tap_beh["Tapping_Times"].min()
        )
    if sub in ["sub-110"]:
        onsets_photo = onsets_photo[1::]
        tap_beh = tap_beh.loc[5::]
        tap_beh["Tapping_Times"] = tap_beh["Tapping_Times"] - (
            tap_beh["Tapping_Times"].min() - onsets_photo[0]
        )

    if len(onsets_photo) != 420 and sub not in [
        "sub-01",
        "sub-02",
        "sub-48",
        "sub-57",
        "sub-80",
        "sub-91",
        "sub-93",
        "sub-99",
        "sub-109",  # Recording started late (less trials in photo)
        "sub-110",  # Recording started late (less trials in photo)
    ]:
        plt.vlines(onsets_photo, 0, 1, color="blue")
        plt.vlines(tap_beh["Tapping_Times"].values + onsets_photo[0], 1, 2, color="red")
        raise ValueError(
            f"    - WARNING: Number of events is not 420 ({len(onsets_photo)})"
        )

    # Correct for delay between photo and behavioral data
    onsets_beh = tap_beh["Tapping_Times"].values + onsets_photo[0]

    # Compute correlation
    if scipy.stats.pearsonr(onsets_photo, onsets_beh)[0] < 0.999:
        # plt.scatter(onsets_beh, onsets_photo)
        print(f"    - WARNING: Correlation between photo and beh onsets is low.")

    # Preprocess physio
    bio = (
        tap.to_data_frame().bfill().ffill()
    )  # Backfill missing values at the beginning
    bio, info = nk.bio_process(
        ecg=bio["ECG"].values,
        rsp=bio["RSP"].values,
        sampling_rate=tap.info["sfreq"],
    )

    # QC
    qc_tap_ecg = qc_physio(bio, info, sub, plot_ecg=qc_tap_ecg)

    # Epoch around each tap
    epochs = nk.epochs_create(
        bio,
        onsets_beh,  # onsets_photo
        sampling_rate=tap.info["sfreq"],
        epochs_start=-4,
        epochs_end=4,
    )

    # Closest R-peak to each tap
    dat = pd.DataFrame()  # Initialize dataframe
    for _, e in epochs.items():
        # Filter df at 0
        at_index = e.iloc[np.argmin(np.abs(e.index)), :]
        # R-peaks
        rpeaks = e[e["ECG_R_Peaks"] == 1].index.values
        r = np.nan if len(rpeaks) == 0 else rpeaks[np.argmin(np.abs(rpeaks))]
        r_pre = np.nan if sum(rpeaks < 0) == 0 else np.max(rpeaks[rpeaks < 0])
        r_post = np.nan if sum(rpeaks >= 0) == 0 else np.min(rpeaks[rpeaks >= 0])
        # RSP - peaks
        p = e[e["RSP_Peaks"] == 1].index.values
        rsp_peak = np.nan if len(p) == 0 else p[np.argmin(np.abs(p))]
        rsp_pre = np.nan if sum(p < 0) == 0 else np.max(p[p < 0])
        rsp_post = np.nan if sum(p >= 0) == 0 else np.min(p[p >= 0])
        # RSP - troughs
        t = e[e["RSP_Troughs"] == 1].index.values
        rsp_trough = np.nan if len(t) == 0 else t[np.argmin(np.abs(t))]
        rsp_trough_pre = np.nan if sum(t < 0) == 0 else np.max(t[t < 0])
        rsp_trough_post = np.nan if sum(t >= 0) == 0 else np.min(t[t >= 0])

        dat = pd.concat(
            [
                dat,
                pd.DataFrame(
                    {
                        "ECG_Rate": at_index["ECG_Rate"],
                        "Closest_R": r,
                        "Closest_R_Pre": r_pre,
                        "Closest_R_Post": r_post,
                        "ECG_Phase_Atrial": at_index["ECG_Phase_Atrial"],
                        "ECG_Phase_Ventricular": at_index["ECG_Phase_Ventricular"],
                        "Closest_RSP_Peak": rsp_peak,
                        "Closest_RSP_Peak_Pre": rsp_pre,
                        "Closest_RSP_Peak_Post": rsp_post,
                        "Closest_RSP_Trough": rsp_trough,
                        "Closest_RSP_Trough_Pre": rsp_trough_pre,
                        "Closest_RSP_Trough_Post": rsp_trough_post,
                        "RSP_Phase": at_index["RSP_Phase"],
                        "RSP_Phase_Completion": at_index["RSP_Phase_Completion"],
                    },
                    index=[0],
                ),
            ],
            axis=0,
        )

    # TODO: HEP amplitude of closest (pre) R-peak

    # Compute tapping rate
    tap_beh["Tapping_Rate"] = np.nan
    for cond in tap_beh["Condition"].unique():
        tap_times = tap_beh[tap_beh["Condition"] == cond]["Tapping_Times"].values
        np.diff(tap_times)
        rate = nk.signal_rate(
            tap_times,
            sampling_rate=1000,
            show=False,
            interpolation_method="monotone_cubic",
        )
        tap_beh.loc[tap_beh["Condition"] == cond, "Tapping_Rate"] = rate

    # Merge with behavioral data
    dat = pd.concat(
        [tap_beh.reset_index(drop=True), dat.reset_index(drop=True)], axis=1
    )
    return dat, qc_tap_ecg


# Variables ==================================================================================
# Change the path to your local data folder.
# The data can be downloaded from OpenNeuro (TODO).
path = "C:/Users/domma/Box/Data/InteroceptionPrimals/Reality Bending Lab - InteroceptionPrimals/"
# path = "C:/Users/dmm56/Box/Data/InteroceptionPrimals/Reality Bending Lab - InteroceptionPrimals/"

# Get participant list
meta = pd.read_csv(path + "participants.tsv", sep="\t")

# Initialize variables
df = pd.DataFrame()
df_tap = pd.DataFrame()

qc_tap_ecg = []

# sub = "sub-50"
# Loop through participants ==================================================================
for i, sub in enumerate(meta["participant_id"].values):

    # if i < 22:
    #     continue

    # Print progress and comments
    print(sub)
    print("  * " + meta[meta["participant_id"] == sub]["Comments"].values[0])

    # Tapping Task ===========================================================================
    print("  - TAP - Preprocessing")

    if sub in ["sub-09"]:
        print("    - WARNING: Skipping TAP for this participant (No ECG).")
    elif sub in ["sub-31"]:
        print("    - WARNING: Skipping TAP for this participant (ECG electrode fell).")
    elif sub in ["sub-49", "sub-55"]:
        print(
            "    - WARNING: Skipping TAP for this participant (no photosensor - unclear why)."
        )
    elif sub in ["sub-68", "sub-72", "sub-75"]:
        print("    - WARNING: Skipping TAP for this participant (technical issues).")
    elif sub in ["sub-114"]:
        print("    - WARNING: Skipping TAP for this participant (technical issues).")
    else:
        tap, qc_tap_ecg = process_tap(sub, qc_tap_ecg)
        df_tap = pd.concat([df_tap, tap], axis=0)


# # Clean up and Save data
df_tap.to_csv("../data/rawdata_tap.csv", index=False)

# Save figures
ill.image_mosaic(qc_tap_ecg, ncols=4, nrows="auto").save(
    "figures/signals_qc_tap_ecg.png"
)

print("Done!")
