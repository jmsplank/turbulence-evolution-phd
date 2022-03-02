import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import os_shortcuts, override_mpl
from phdhelper.helpers.CONSTANTS import R_e
from phdhelper.helpers.COLOURS import red, green, blue
from os.path import join
import json
from datetime import datetime as dt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

# import pybowshock as pybs

override_mpl.override()

gen_path = lambda x: os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."), x)
path_13 = gen_path("20180313")
path_16 = gen_path("20180316")
path_18 = gen_path("20200318")

fgm = "data/fgm"
mag = join(fgm, "data.npy")
mag_time = join(fgm, "time.npy")

fpi = "data/fpi"
vi = join(fpi, "data_bulkv_i.npy")
vi_time = join(fpi, "time_bulkv_i.npy")
ndi = join(fpi, "data_numberdensity_i.npy")
ndi_time = join(fpi, "time_numberdensity_i.npy")
nde = join(fpi, "data_numberdensity_e.npy")
nde_time = join(fpi, "time_numberdensity_e.npy")
espec_i = join(fpi, "data_energyspectr_omni_i.npy")
espec_i_time = join(fpi, "time_energyspectr_omni_i.npy")
espec_i_bins = join(fpi, "bins_energyspectr_omni_i.npy")

rows = 5
fig, ax = plt.subplots(
    rows, 4, figsize=(10, 10 * (9 / 16)), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]}
)

# for shock, path in enumerate([path_13, path_16, path_18]):
#     B_time = np.load(path(mag_time))
#     print(
#         f"& {dt.utcfromtimestamp(B_time[0]):%Y/%m/%d %H:%M:%S} & {dt.utcfromtimestamp(B_time[-1]):%Y/%m/%d %H:%M:%S}"
#     )

for shock, path in enumerate([path_13, path_16, path_18]):
    B = np.load(path(mag))
    B_time = np.load(path(mag_time))

    print(B_time[-1] - B_time[0])

    V_i = np.load(path(vi))
    V_i_time = np.load(path(vi_time))

    Nd_i = np.load(path(ndi))
    Nd_e = np.load(path(nde))
    Nd_i_time = np.load(path(ndi_time))
    Nd_e_time = np.load(path(nde_time))

    Espec_i = np.load(path(espec_i))
    Espec_i[Espec_i == 0] = np.min(Espec_i[Espec_i != 0])
    Espec_i_time = np.load(path(espec_i_time))
    Espec_i_bins = np.load(path(espec_i_bins))[0, :]

    ax[0, shock].plot(B_time, B[:, 3], color="k")
    for i in range(3):
        ax[1, shock].plot(
            B_time, B[:, i], label=[f"GSE {x}" for x in ["X", "Y", "Z"]][i]
        )
        ax[2, shock].plot(
            V_i_time, V_i[:, i], label=[f"GSE {x}" for x in ["X", "Y", "Z"]][i]
        )
    ax[3, shock].plot(Nd_i_time, Nd_i, label="$n_i$ $[cm^{-3}]$")
    ax[3, shock].plot(Nd_e_time, Nd_e, label="$n_e$ $[cm^{-3}]$")

    im = ax[4, shock].pcolormesh(
        Espec_i_time[::10],
        Espec_i_bins,
        Espec_i.T[:, ::10],
        cmap="viridis",
        norm=LogNorm(),
        shading="nearest",
    )
    ax[4, shock].set_yscale("log")

    ax[1, shock].get_shared_x_axes().join(ax[0, shock], ax[1, shock])
    ax[2, shock].get_shared_x_axes().join(ax[0, shock], ax[2, shock])
    ax[3, shock].get_shared_x_axes().join(ax[0, shock], ax[3, shock])
    ax[4, shock].get_shared_x_axes().join(ax[0, shock], ax[4, shock])

    quarters = lambda frac: B_time[int(len(B_time) * frac)]
    fmt = lambda t: dt.strftime(dt.utcfromtimestamp(t), "%H:%M")
    values = np.arange(int(B_time[0]), int(B_time[-1]), 1)
    values = values[values % 300 == 0]
    for i in range(rows):
        ax[i, shock].set_xticks(values)
        ax[i, shock].xaxis.set_minor_locator(MultipleLocator(60))
        # ax[i, shock].grid(None)

    ax[-1, shock].set_xticklabels([fmt(lab) for lab in values])

    ax[0, shock].set_xticklabels([])
    ax[1, shock].set_xticklabels([])
    ax[2, shock].set_xticklabels([])
    ax[3, shock].set_xticklabels([])

for i in range(rows):
    ax[i, 1].sharey(ax[i, 0])
    ax[i, 2].sharey(ax[i, 1])

    plt.setp(ax[i, 1].get_yticklabels(), visible=False)
    plt.setp(ax[i, 2].get_yticklabels(), visible=False)

with open(path_13("summary.json"), "r") as file:
    sum13 = json.load(file)
for i in range(rows):
    ax[i, 0].axvline(sum13["example_plot_time"]["timestamp"], color="k")

for i in range(rows - 1):
    ax[i, -1].axis("off")

cbar = plt.colorbar(im, cax=ax[-1, -1])
cbar.set_ticks([1e4, 1e6, 1e8])
cbar.set_label("$DEF$  $\left[\\frac{keV}{cm^2\, s\, sr\, keV}\\right]$")

ax[0, 0].set_ylim((0, 50))
ax[1, 0].set_ylim((-45, 45))
ax[2, 0].set_ylim((-700, 300))
ax[3, 0].set_ylim((-10, 110))

ax[4, 0].set_yticks(np.logspace(1, 4, 4))
ax[4, 0].set_yticklabels([f"$10^{i}$" for i in range(1, 5)])

ax[0, 0].set_ylabel("$|B|$ [$nT$]")
ax[1, 0].set_ylabel("$B$ [$nT$]")
ax[2, 0].set_ylabel("$v_i$ [$kms^{-1}$]")
ax[3, 0].set_ylabel("$n$ [$cm^{-3}$]")
ax[4, 0].set_ylabel("$E_i$ $[eV]$")

ax[-1, 0].set_xlabel("13 Mar 2018")
ax[-1, 1].set_xlabel("16 Mar 2018")
ax[-1, 2].set_xlabel("18 Mar 2020")

ax[0, 0].set_title("A")
ax[0, 1].set_title("B")
ax[0, 2].set_title("C")

ax[1, -2].legend(fontsize=8, loc="upper right")
ax[2, -2].legend(fontsize=8, loc="upper right")
ax[3, -2].legend(fontsize=8, loc="upper right")

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0)
plt.savefig(join(os_shortcuts.get_path(__file__), "summary_plot.pdf"), dpi=300)
plt.savefig(join(os_shortcuts.get_path(__file__), "summary_plot.png"), dpi=300)
plt.show()
