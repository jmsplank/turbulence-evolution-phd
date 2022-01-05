import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import os_shortcuts, override_mpl
from phdhelper.helpers.CONSTANTS import R_e
from phdhelper.helpers.COLOURS import red, green, blue, mandarin
from os.path import join
import json
import pybowshock as pybs

override_mpl.override()

gen_path = lambda x: os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."), x)
path_13 = gen_path("20180313")
path_16 = gen_path("20180316")
path_18 = gen_path("20200318")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
# ax = [ax]

for i, path in enumerate([path_16, path_18]):
    slopes = np.load(path("mag_spec/slope_interp.npy"))
    k = np.load(path("mag_spec/x_interp.npy"))
    k = 10 ** k
    with open(path("summary.json"), "r") as file:
        summary = json.load(file)

    times = np.load(path("mag_spec/times.npy"))
    print(times[1] - times[0])
    sw = summary["SW"]["timestamp"]

    sections = summary["sections"]["timestamp"]  # Get the sections to split into
    arg = lambda x: np.argmin(np.abs(x - times))
    sections = np.array([times[arg(x)] for x in sections])  # Roud to window time

    sections = np.insert(sections, 0, times[0])  # Insert first window at index 0
    sections = np.append(sections, 20)  # Append last window

    ax[i].axhline(-5 / 3, color="k", alpha=0.5, label="$-5/3$")

    for region in range(len(sections) - 1):
        slope = slopes[(times >= sections[region]) & (times < sections[region + 1]), :]
        if region == len(sections) - 2:  # To include final window
            slope = slopes[times >= sections[region], :]

        length = slope.shape[0]
        print(f"{summary['theta_Bn']} : {length=}")

        ax[i].fill_between(
            k,
            slope.mean(axis=0) - (slope.std(axis=0) / np.sqrt(length)),
            slope.mean(axis=0) + (slope.std(axis=0) / np.sqrt(length)),
            color=[red, green, blue, mandarin][4 - (len(sections) - 1) :][region],
            alpha=0.2,
            lw=0,
            step="mid",
        )
        ax[i].step(
            k,
            slope.mean(axis=0),
            label=summary["sections"]["label"][region],
            where="mid",
            color=[green, blue, mandarin][region]
            if i == 0
            else [blue, mandarin][region],
        )

    rhoi = ax[i].axvline(
        10 ** [-2.25, -2.5][i],
        ls="-.",
        label=r"$1/\rho_i$",
        color="k",
    )
    di = ax[i].axvline(
        10 ** [-1.85, -1.95][i],
        ls="--",
        label=r"$1/d_i$",
        color="k",
    )
    rhod = ax[i].axvline(
        10 ** [-0.15, -0.25][i],
        ls="-",
        label=r"$1/d_e\approx1/\rho_e$",
        color="k",
    )

    ax[i].set_xscale("log")
    ax[i].legend(loc="lower left", fontsize=6)
    ax[i].set_ylabel(rf"Slope ($\theta_{{Bn}}={summary['theta_Bn']:02.0f}^\circ$)")
    ax[i].grid(False)
    ax[i].set_ylim((-7.8, 0.5))
    ax[i].set_xlim((k[0], k[-1]))

ax[-1].set_xlabel("$k$ $[km^{-1}]$")

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(gen_path("plots")("Appendix_slope_avg.pdf"))
plt.show()
