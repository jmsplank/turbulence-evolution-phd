"""
Last modified:  19/04/21

Moving windows of magnetic spectrum, with variable ion & electron limits.

"""

import os
from sys import stderr
import numpy as np
import matplotlib.pyplot as plt
import logging as log
import time as timetime
import json
from scipy.optimize import curve_fit
from scipy import interpolate
from tqdm import tqdm
from phdhelper.helpers import override_mpl
import pandas as pd
import subprocess
from datetime import datetime as dt
from phdhelper.helpers.COLOURS import red, green, blue, mandarin
from matplotlib.ticker import MaxNLocator

override_mpl.override("|krgb")
override_mpl.cmaps(name="custom_diverging")


path = os.path.dirname(os.path.realpath(__file__))
dirpath = "/".join(path.split("/")[:-1])
print(dirpath)

log.basicConfig(
    filename=f"{path}/mag_spec.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def extract_grads(k, y, ion, elec, instr):
    def split(X, Y, low, high):
        mask_x = X[(X >= low) & (X <= high)]
        mask_y = Y[(X >= low) & (X <= high)]

        return (np.log10(mask_x), np.log10(mask_y))

    def fit(X, Y):
        grad, pcov = curve_fit(lambda x, m, c: c + m * x, X, Y)
        return grad[0]

    # Iner
    a, b = split(k, y, k[0], ion)
    iner = fit(a, b)

    # Ion
    a, b = split(k, y, ion, elec)
    ion = fit(a, b)

    # Elec
    a, b = split(k, y, elec, instr)
    elec = fit(a, b)

    return np.array([iner, ion, elec])


def lengths(s, number_density, temp_perp, B_field, all=False):
    if s == "i":
        log.info("---->----IONS----<----")
    else:
        log.info("---->----ELEC----<----")

    n = number_density.mean()
    const = 1.32e3 if s == "i" else 5.64e4
    # https://en.wikipedia.org/wiki/Plasma_parameters#Fundamental_plasma_parameters
    omega_p = const * np.sqrt(n)
    p = 2.99792458e8 / omega_p
    p /= 1e3
    log.info(f"Inertial length: {p:.3f}km")

    T = temp_perp
    v = (
        np.sqrt(
            np.mean(T)
            * 2
            * 1.60217662e-19
            / (1.6726219e-27 if s == "i" else 9.10938356e-31)
        )
        / 1e3
    )
    B_scaled = B_field.copy() * 1e-9
    BT = np.linalg.norm(B_scaled, axis=1).mean()
    log.info(f"V: {v:.3f}kms⁻¹")
    omega_c = 1.60217662e-19 * BT / (1.6726219e-27 if s == "i" else 9.10938356e-31)
    rho = v / omega_c
    log.info(f"Gyroradius: {rho:.3f}km")
    log.info("---->----<<>>----<----")
    log.info("")
    # return limit

    log.info(
        f"n: {n} | const: {const} | omega_p: {omega_p} | v: {v} | BT: {BT} | omega_c: {omega_c}"
    )

    if all:
        return np.array([rho, p])
    else:
        if s == "i":
            return rho
        else:
            return p


log.info("Loading data")
big_data = np.load(f"{dirpath}/data/fsm/data.npy")
log.info("Loading time")
big_time = np.load(f"{dirpath}/data/fsm/time.npy")
# td = big_time[1] - big_time[0]
td = 1 / 8192

log.info("Loading temp_perp")
big_temp_perp_e = np.load(f"{dirpath}/data/fpi/data_tempperp_e.npy")
big_temp_perp_i = np.load(f"{dirpath}/data/fpi/data_tempperp_i.npy")
log.info("Loading number_density")
big_number_density_e = np.load(f"{dirpath}/data/fpi/data_numberdensity_e.npy")
big_number_density_i = np.load(f"{dirpath}/data/fpi/data_numberdensity_i.npy")
log.info("Loading electron tempperp time")
time_tempperp_e = np.load(f"{dirpath}/data/fpi/time_tempperp_e.npy")
log.info("Loading ion tempperp time")
time_tempperp_i = np.load(f"{dirpath}/data/fpi/time_tempperp_i.npy")
log.info("Loading electron numberdensity time")
time_numberdensity_e = np.load(f"{dirpath}/data/fpi/time_numberdensity_e.npy")
log.info("Loading ion numberdensity time")
time_numberdensity_i = np.load(f"{dirpath}/data/fpi/time_numberdensity_i.npy")
log.info("Loading stats")
with open(f"{dirpath}/data/fpi/stats.json") as f:
    stats = json.load(f)
meanv = stats["mean_v"]["value"]

N = 100  # Number of windows

ion_lim = 1.0 / lengths("i", big_number_density_i, big_temp_perp_i, big_data)
electron_lim = 1.0 / lengths("e", big_number_density_e, big_temp_perp_e, big_data)

min_freq = ion_lim * 5
# bin_size = int(1 / (min_freq * td))
bin_size = int(9 / td)

max_index = len(big_data) - bin_size
log.info(f"max index: {max_index}")
bin_starts = np.arange(0, max_index, bin_size, dtype=int)
print(f"There are {len(bin_starts)} windows")

grads = []
times = []
knots = []
slope_lims = []
slope_lims_other = []
slope_interp = []
spectra = []
fsm = []

for bin in tqdm(bin_starts):
    Y = {}

    data = big_data[bin : bin + bin_size, :]
    time = big_time[bin : bin + bin_size]

    fsm.append(np.linalg.norm(data, axis=1).mean())

    # log.info("Comutping FFT over each coord")
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

    # Scale by kolmogorov
    y = y * (k ** (5 / 3))

    number_density_i = big_number_density_i[
        (time_numberdensity_i >= time[0]) & (time_numberdensity_i <= time[-1])
    ]
    temp_perp_i = big_temp_perp_i[
        (time_tempperp_i >= time[0]) & (time_tempperp_i <= time[-1])
    ]

    number_density_e = big_number_density_e[
        (time_numberdensity_e >= time[0]) & (time_numberdensity_e <= time[-1])
    ]
    temp_perp_e = big_temp_perp_e[
        (time_tempperp_e >= time[0]) & (time_tempperp_e <= time[-1])
    ]

    ion_lims = 1.0 / lengths("i", number_density_i, temp_perp_i, data, all=True)
    electron_lims = 1.0 / lengths("e", number_density_e, temp_perp_e, data, all=True)

    ion_lim = ion_lims[0]
    electron_lim = electron_lims[1]

    # grads.append(extract_grads(k, y, ion_lim, electron_lim, 10))
    times.append(big_time[bin + bin_size // 2])

    instrument_mask = k <= 10
    kk = np.log10(k[instrument_mask])
    yy = np.log10(y[instrument_mask])

    f = interpolate.interp1d(kk, yy)
    xx = np.log10(np.logspace(kk[0], kk[-1], num=1000))
    yy = f(xx)

    INTERP_MIN = min(kk)
    # print(10 ** INTERP_MIN / (2 * np.pi / meanv))
    INTERP_MAX = max(kk)
    x_interp = np.linspace(
        INTERP_MIN,
        INTERP_MAX,
        32,
    )

    spectra.append(10 ** f(x_interp))

    r_data = {"x": xx, "y": yy}
    r_df = pd.DataFrame(r_data)
    r_df.to_csv(f"{path}/raw_r.csv")

    devnull = open(os.devnull, "w")
    # subprocess.call(["Rscript", f"{path}/mars.r"])  # Debug show R output
    subprocess.call(["Rscript", f"{path}/mars.r"], stdout=devnull, stderr=devnull)
    r_out = pd.read_csv(f"{path}/mars.csv")
    YY = np.array(r_out.y)
    slopes_all = np.gradient(YY, abs(xx[0] - xx[1]))

    # Correct for scaling
    slopes_all = slopes_all - (5 / 3)

    slopes, slope_index, slope_counts = np.unique(
        np.round(slopes_all, 2),
        return_index=True,
        return_counts=True,
    )
    slopes_all = np.round(slopes_all, 2)
    slope_counts = slope_counts > 10
    slopes = slopes[slope_counts]
    slope_index = slope_index[slope_counts]
    slope_k = 10 ** xx[slope_index]

    f = interpolate.interp1d(xx, slopes_all, fill_value=np.nan)
    interp_slope = f(x_interp)
    slope_interp.append(interp_slope)

    slope_lims.append(np.log10(np.array([ion_lim, electron_lim])))
    slope_lims_other.append(np.log10(np.array([ion_lims[1], electron_lims[0]])))

    ks = np.histogram(
        np.log10(slope_k),
        bins=x_interp,
    )[0]
    ks = ks.astype("float")
    ks[ks == 0] = np.nan
    knots.append(ks)

    # Unscale by kolmogorov
    y = y / (k ** (5 / 3))

    example_bin = 4054400
    if bin == example_bin:
        # example_ts = int(big_time[example_bin])
        # example_dt = f"{dt.utcfromtimestamp(example_ts):%Y/%m/%d %H:%M:%S}"
        # with open(f"{dirpath}/summary.json", "r") as file:
        #     summary = json.load(file)
        #     summary["example_plot_time"]["timestamp"] = example_ts
        #     summary["example_plot_time"]["datetime"] = example_dt
        # with open(f"{dirpath}/summary.json", "w") as file:
        #     json.dump(
        #         summary,
        #         file,
        #         indent=4,
        #         sort_keys=True,
        #         separators=(", ", ": "),
        #         ensure_ascii=False,
        #     )

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.loglog(k, y, color="k", label="Magnetic spectrum")
        xx = 10 ** xx
        YY = 10 ** YY
        ax.loglog(xx, YY / (xx ** (5 / 3)), color=mandarin, ls="--", label="MARS fit")
        lims = (1e-16, 1)
        for ks in range(len(slope_k)):
            ax.axvline(slope_k[ks], color=mandarin, alpha=0.4)
        ax.axvline(ion_lim, color=red, label=r"$\rho_i$")
        ax.axvline(ion_lims[1], color=red, ls="--", label=r"$d_i$")
        ax.axvline(electron_lim, color=green, label=r"$\rho_e$")
        ax.axvspan(10, k[-1], fc="k", ec=None, alpha=0.1, label="Noise")
        ax.grid(False)

        ax.set_ylim(lims)
        ax.set_xlim((k[0], k[-1]))

        ax.set_ylabel(r"Magnetic spectrum [$nT^2Hz^{-1}$]")
        ax.set_xlabel("Wavenumber $k\,[km^{-1}]$")
        ax.set_title(f"{dt.utcfromtimestamp(times[-1]):%Y/%m/%d %H:%M:%S}")

        def ref_line(x, n):
            return (-5 / 3 * np.log10(x)) + (n + 5 / 3 * np.log10(min(x)))

        yt = np.log10(ax.get_yticks())
        ax.plot(
            xx,
            10 ** ref_line(xx, yt[0]),
            color="#E6E6E6",
            zorder=-5,
            label="$-5/3$ slope",
        )
        for i in yt[::2][1:]:
            im = ax.plot(xx, 10 ** ref_line(xx, i), color="#E6E6E6", zorder=-5)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f"{path}/example_{bin}_{int(times[-1])}.png", dpi=300)
        plt.savefig(f"{path}/example_{bin}_{int(times[-1])}.pdf", dpi=300)
        # plt.show()
        plt.clf()
        plt.close()
        del fig, ax

    del y
    del Y
    del YY
    del freq
    del B
    del Hann
    del Yi
    del data
    del time

grads = np.array(grads)
knots = np.array(knots)
slope_lims = np.array(slope_lims)
slope_lims_other = np.array(slope_lims_other)
slope_interp = np.array(slope_interp)
fsm = np.array(fsm)
np.save(f"{path}/grads.npy", grads)
np.save(f"{path}/times.npy", times)
np.save(f"{path}/knots.npy", np.array(knots))
np.save(f"{path}/slope_interp.npy", np.array(slope_interp))
np.save(f"{path}/spectra.npy", np.array(spectra))
np.save(f"{path}/fsm_sampled_100.npy", fsm)
np.save(f"{path}/x_interp.npy", x_interp)

fig, ax = plt.subplots(
    2,
    2,
    gridspec_kw={"width_ratios": [98, 3], "height_ratios": [30, 70]},
    figsize=(6, 4),
)

ax1 = ax[0, 0]
ax_main = ax[1, 0]
cbar_main = ax[1, 1]
ax[0, 1].axis("off")

ax1.plot(big_time, np.linalg.norm(big_data, axis=1))

X = np.linspace(big_time[0], big_time[bin_starts[-1] + bin_size], len(bin_starts) + 1)
Y = np.logspace(INTERP_MIN, INTERP_MAX, 33)
X, Y = np.meshgrid(X, Y)

im = ax_main.pcolor(
    X,
    Y,
    slope_interp.T,
    vmin=-4.667,
    vmax=1.333,
    cmap="custom_diverging",
)
fig.colorbar(im, cax=cbar_main)
ax_main.set_yscale("log")

slope_lims[slope_lims == 0] = np.nan
slope_lims_other[slope_lims_other == 0] = np.nan
ax_main.plot(
    times,
    10 ** slope_lims[:, 0],
    label=r"$1/\rho_i$",
    color="k",
    lw=1,
    ls="-.",
)
ax_main.plot(
    times,
    10 ** slope_lims_other[:, 0],
    label="$1/d_i$",
    ls="--",
    color="k",
    lw=1,
)
ax_main.plot(
    times,
    10 ** slope_lims[:, 1],
    label=r"$1/\rho_e\approx 1/d_e$",
    color="k",
    lw=1,
)

ax_main.legend(loc="upper left", fontsize=8)

ax1.tick_params(
    axis="x",
    which="both",
    bottom="on",
    top="on",
    labelbottom=False,
)

ax1.set_ylabel("$|B|$ [$nT$]")
ax_main.set_ylabel(r"$k\quad[km^{-1}]$")
cbar_main.set_ylabel("Slope")
ax_main.set_xlabel(
    f"Time UTC {dt.strftime(dt.utcfromtimestamp(big_time[0]), r'%d/%m/%Y')} (hh:mm:ss)"
)

ticks = np.arange(int(big_time[0]), big_time[-1], dtype=int)
ticks = ticks[ticks % 300 == 0]
for ax in [ax1, ax_main]:
    ax.set_xlim((big_time[0], big_time[-1]))
    ax.set_xticks(ticks)
ax_main.set_xticklabels([f"{dt.utcfromtimestamp(a):%H:%M:%S}" for a in ticks])

ax1.get_shared_x_axes().join(ax1, ax_main)

ax1.yaxis.set_major_locator(MaxNLocator(prune="lower", nbins=4))
ax_main.set_ylim((10 ** INTERP_MIN, 10 ** INTERP_MAX))

plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0)
plt.savefig(f"{path}/{dt.utcfromtimestamp(big_time[0]):%Y%m%d}_slopes.png", dpi=300)
plt.savefig(f"{path}/{dt.utcfromtimestamp(big_time[0]):%Y%m%d}_slopes.pdf", dpi=300)
# plt.show()
