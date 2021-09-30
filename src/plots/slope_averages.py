import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import os_shortcuts, override_mpl
from phdhelper.helpers.CONSTANTS import R_e
from phdhelper.helpers.COLOURS import red, green, blue
from os.path import join
import json
import pybowshock as pybs

override_mpl.override()

gen_path = lambda x: os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."), x)
path_13 = gen_path("20180313")
path_16 = gen_path("20180316")
path_18 = gen_path("20200318")

fig, ax = plt.subplots(3, 1, sharex=True)

for i, path in enumerate([path_13, path_16, path_18]):
    slopes = np.load(path("mag_spec/slope_interp.npy"))
    k = np.load(path_18("mag_spec/x_interp.npy"))
    with open(path("summary.json"), "r") as file:
        summary = json.load(file)
    times = np.load(path("mag_spec/times.npy"))
    sw = summary["SW"]["timestamp"]

    in_sw = slopes[times > sw, :].shape[0]
    out_sw = slopes[times < sw, :].shape[0]
    print(f"{in_sw = } | {out_sw = }")

    k = 10 ** k

    ax[i].axhline(-5 / 3, color="k", alpha=0.5, label="$-5/3$")

    ax[i].fill_between(
        k,
        slopes[times < sw, :].mean(axis=0)
        - (slopes[times < sw, :].std(axis=0) / np.sqrt(out_sw)),
        slopes[times < sw, :].mean(axis=0)
        + (slopes[times < sw, :].std(axis=0) / np.sqrt(out_sw)),
        color=red,
        alpha=0.2,
        lw=0,
        step="mid",
    )
    ax[i].fill_between(
        k,
        slopes[times > sw, :].mean(axis=0)
        - (slopes[times > sw, :].std(axis=0) / np.sqrt(in_sw)),
        slopes[times > sw, :].mean(axis=0)
        + (slopes[times > sw, :].std(axis=0) / np.sqrt(in_sw)),
        color=green,
        alpha=0.2,
        lw=0,
        step="mid",
    )

    ax[i].step(k, slopes[times < sw, :].mean(axis=0), label=f"STR", where="mid")
    ax[i].step(k, slopes[times > sw, :].mean(axis=0), label=f"SW", where="mid")

    ax[i].set_xscale("log")
    ax[i].legend(loc="lower left", fontsize=8)
    ax[i].set_ylabel(rf"Slope ($\theta_{{Bn}}={summary['theta_Bn']:02.0f}^\circ$)")
    ax[i].grid(False)
    ax[i].set_ylim((-4.5, 0.5))

ax[-1].set_xlabel("$k$ $[km^{-1}]$")

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(gen_path("plots")("slope_averages.pdf"))
plt.show()
