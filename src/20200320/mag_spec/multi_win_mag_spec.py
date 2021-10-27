"""20200320 is spread over multiple burst intervals.
As such, code for this event has to be re-written.
"""
import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.physics import lengths
from phdhelper.helpers.COLOURS import red, green
import json
from numpyencoder import NumpyEncoder
from scipy.interpolate import interp1d
import pandas as pd
import os
import subprocess
from tqdm import tqdm

from scipy.optimize.optimize import main

override_mpl.override()
override_mpl.cmaps(name="custom_diverging")

fsm_path = new_path(get_path(__file__, ".."), "data/fsm")
all_b = np.load(fsm_path("data.npy"))
all_b_time = np.load(fsm_path("time.npy"))

main_path = new_path(get_path(__file__, ".."))

fpi_path = new_path(get_path(__file__, ".."), "data/fpi")
all_n_i = np.load(fpi_path("data_numberdensity_i.npy"))
all_n_i_time = np.load(fpi_path("time_numberdensity_i.npy"))
all_n_e = np.load(fpi_path("data_numberdensity_e.npy"))
all_n_e_time = np.load(fpi_path("time_numberdensity_e.npy"))
all_temp_i = np.load(fpi_path("data_tempperp_i.npy"))
all_temp_i_time = np.load(fpi_path("time_tempperp_i.npy"))
all_temp_e = np.load(fpi_path("data_tempperp_e.npy"))
all_temp_e_time = np.load(fpi_path("time_tempperp_e.npy"))

SPACING = 9.5  # seconds of data for each window
td = all_b_time[1] - all_b_time[0]

with open(main_path("summary.json"), "r") as file:
    summary = json.load(file)

with open(fpi_path("stats.json"), "r") as file:
    stats = json.load(file)

meanv = stats["mean_v"]["value"]

burst_start = summary["burst_start"]
burst_stop = summary["burst_stop"]

slopes_all = []
times_all = []
k_extent_all = []


for burst in range(len(burst_start)):
    burst_b_time = all_b_time[burst_start[burst] : burst_stop[burst]]
    windows = np.arange(burst_b_time[0], burst_b_time[-1], SPACING)
    print(f"Burst {burst+1} has {len(windows)} windows.")
    slopes_burst = []
    times_burst = []
    for win in tqdm(range(len(windows) - 1)):
        data = all_b[(all_b_time >= windows[win]) & (all_b_time < windows[win + 1]), :]
        time = all_b_time[
            (all_b_time >= windows[win]) & (all_b_time < windows[win + 1])
        ]
        Y = {}
        for i in range(3):
            # log.info(f"index {i}")
            B = data[:, i] * 1e-9
            # log.info("Scaling mean")
            B -= B.mean()

            # log.info("Applying Hanning window")
            Hann = np.hanning(len(B)) * B
            # log.info("Calculating FFT")
            Yi = np.fft.fft(Hann)
            # log.info("Calculating Frequencies")
            freq = np.fft.fftfreq(len(B), td)
            # log.info("Obtaining power spectrum")
            Y[["x", "y", "z"][i]] = (np.power(np.abs(Yi), 2) * 1e9 * td)[freq > 0]
        # log.info("Summing components")
        y = np.sum([Y[i] for i in ["x", "y", "z"]], axis=0)
        k = freq[freq > 0] * 2 * np.pi / meanv

        n_i = all_n_i[(all_n_i_time >= time[0]) & (all_n_i_time <= time[-1])]
        n_e = all_n_e[(all_n_e_time >= time[0]) & (all_n_e_time <= time[-1])]
        temp_i = all_temp_i[
            (all_temp_i_time >= time[0]) & (all_temp_i_time <= time[-1])
        ]
        temp_e = all_temp_e[
            (all_temp_e_time >= time[0]) & (all_temp_e_time <= time[-1])
        ]

        ion_lims = 1.0 / lengths(
            "i",
            all=True,
            number_density=n_i,
            temp_perp=temp_i,
            B_field=data,
        )
        electron_lims = 1.0 / lengths(
            "e",
            all=True,
            number_density=n_e,
            temp_perp=temp_e,
            B_field=data,
        )

        instrument_mask = k <= 10
        kk = np.log10(k[instrument_mask])
        yy = np.log10(y[instrument_mask])

        f = interp1d(kk, yy)
        xx = np.log10(np.logspace(kk[0], kk[-1], num=1000))
        yy = f(xx)

        INTERP_MIN = min(kk)
        INTERP_MAX = max(kk)
        x_interp = np.linspace(
            INTERP_MIN,
            INTERP_MAX,
            32,
        )
        k_extent_burst = [INTERP_MIN, INTERP_MAX]

        r_data = {"x": xx, "y": yy}
        r_df = pd.DataFrame(r_data)
        r_df.to_csv(main_path("mag_spec/raw_r.csv"))

        with open(os.devnull, "w") as devnull:
            subprocess.call(
                ["Rscript", main_path("mag_spec/mars.r")],
                stdout=devnull,
                stderr=devnull,
            )
        r_out = pd.read_csv(main_path("mag_spec/mars.csv"))
        YY = np.array(r_out.y)
        slopes_MARS = np.gradient(YY, abs(xx[0] - xx[1]))

        slopes, slope_index, slope_counts = np.unique(
            np.round(slopes_MARS, 2),
            return_index=True,
            return_counts=True,
        )
        slopes_MARS = np.round(slopes_MARS, 2)
        slope_counts = slope_counts > 10
        slopes = slopes[slope_counts]
        slope_index = slope_index[slope_counts]
        slope_k = 10 ** xx[slope_index]

        f = interp1d(xx, slopes_MARS, fill_value=np.nan)
        interp_slope = f(x_interp)

        slopes_burst.append(interp_slope)
        times_burst.append((time[0] + time[-1]) / 2)

    slopes_all.append(np.array(slopes_burst))
    times_all.append(np.array(times_burst))
    k_extent_all.append(np.array(k_extent_burst))

np.save(main_path("mag_spec/slopes_all.npy"), np.array(slopes_all))
np.save(main_path("mag_spec/times_all.npy"), np.array(times_all))
np.save(main_path("mag_spec/k_extent_all.npy"), np.array(k_extent_all))

fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [98, 2]})

for burst in range(len(times_all)):
    im = ax[0].imshow(
        slopes_all[burst].T,
        extent=(
            times_all[burst][0],
            times_all[burst][-1],
            k_extent_all[burst][0],
            k_extent_all[burst][1],
        ),
        origin="lower",
        aspect="auto",
        vmin=-4.667,
        vmax=1.333,
        cmap="custom_diverging",
    )
ax[0].set_xlim((all_b_time[0], all_b_time[-1]))
fig.colorbar(im, cax=ax[1])

plt.show()
