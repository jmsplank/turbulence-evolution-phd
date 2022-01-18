import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers import os_shortcuts as oss
from phdhelper.helpers.CONSTANTS import R_e
from phdhelper.helpers.COLOURS import red, green, blue
from os.path import join
import json
from datetime import datetime as dt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

override_mpl.override()

fig, axs = plt.subplots(
    5, 2, figsize=(8, 8 * (9 / 16)), gridspec_kw={"width_ratios": [3, 0.05]}
)
ax = axs.T[0]
ax2 = axs.T[1]

ax[1].get_shared_x_axes().join(ax[0], ax[1])
ax[2].get_shared_x_axes().join(ax[0], ax[2])
ax[3].get_shared_x_axes().join(ax[0], ax[3])
ax[4].get_shared_x_axes().join(ax[0], ax[4])

path = oss.new_path(oss.get_path(__file__, ".."), "20200320")

with open(path("summary.json"), "r") as file:
    summary = json.load(file)

fgm = np.load(path("data/fgm/data.npy"))
fgm_time = np.load(path("data/fgm/time.npy"))

vi = np.load(path("data/fpi/data_bulkv_i.npy"))
vi_time = np.load(path("data/fpi/time_bulkv_i.npy"))

nd_i = np.load(path("data/fpi/data_numberdensity_i.npy"))
nd_i_time = np.load(path("data/fpi/time_numberdensity_i.npy"))
nd_e = np.load(path("data/fpi/data_numberdensity_e.npy"))
nd_e_time = np.load(path("data/fpi/time_numberdensity_e.npy"))

espec_i = np.load(path("data/fpi/data_energyspectr_omni_i.npy"))
espec_i[espec_i == 0] = np.min(espec_i[espec_i != 0])
espec_i_time = np.load(path("data/fpi/time_energyspectr_omni_i.npy"))
espec_i_bins = np.load(path("data/fpi/bins_energyspectr_omni_i.npy"))[0, :]

farg = lambda x0, x1, arr: np.nonzero((arr >= fgm_time[x0]) & (arr <= fgm_time[x1]))

start = summary["burst_start_fgm"]
stop = summary["burst_stop_fgm"]

# print(
#     f"& {dt.utcfromtimestamp(summary['crop'][0]):%Y/%m/%d %H:%M:%S} & {dt.utcfromtimestamp(summary['crop'][1]):%Y/%m/%d %H:%M:%S}"
# )
last = len(start) - 1

for i in range(len(start)):
    ax[0].plot(fgm_time[start[i] : stop[i]], fgm[start[i] : stop[i], 3], color="k")
    for j in range(3):
        ax[1].plot(
            fgm_time[start[i] : stop[i]],
            fgm[start[i] : stop[i], j],
            color=[red, green, blue][j],
            label=[f"GSE {x}" for x in ["X", "Y", "Z"]][j] if i is last else None,
        )
    slc = farg(start[i], stop[i], vi_time)
    for j in range(3):
        ax[2].plot(
            vi_time[slc],
            vi[slc][:, j],
            color=[red, green, blue][j],
            label=[f"GSE {x}" for x in ["X", "Y", "Z"]][j] if i is last else None,
        )

    slc = farg(start[i], stop[i], nd_i_time)
    ax[3].plot(
        nd_i_time[slc],
        nd_i[slc],
        color=red,
        label="$n_i$ $[cm^{-3}]$" if i is last else None,
    )
    slc = farg(start[i], stop[i], nd_e_time)
    ax[3].plot(
        nd_e_time[slc],
        nd_e[slc],
        color=green,
        label="$n_e$ $[cm^{-3}]$" if i is last else None,
    )

    slc = farg(start[i], stop[i], espec_i_time)
    im = ax[4].pcolormesh(
        espec_i_time[slc][::10],
        espec_i_bins,
        espec_i[slc].T[:, ::10],
        cmap="viridis",
        norm=LogNorm(),
        shading="nearest",
    )
    ax[4].set_yscale("log")


crop = summary["crop"]
ax[-1].set_xlim((crop[0], crop[1]))

fmt = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M")
labels = np.arange(int(crop[0]), int(crop[1]))
labels = labels[labels % 600 == 0]
for i in range(5):
    ax[i].set_xticks(labels)
    ax[i].xaxis.set_minor_locator(MultipleLocator(120))
ax[-1].set_xticklabels([fmt(x) for x in labels])

ax[0].set_ylabel("$|B|$ [$nT$]")
ax[1].set_ylabel("$B$ GSE\n[$nT$]")
ax[2].set_ylabel("$v_i$ GSE\n[$kms^{-1}$]")
ax[3].set_ylabel("$n_{i,e}$\n[$cm^{-3}$]")
ax[4].set_ylabel("$eV$")

ax[-1].set_xlabel(f"20 Mar 2020")

ax[0].set_yticks([0, 20, 40, 60])
ax[0].set_ylim((-5, 65))
ax[3].set_yticks([0, 25, 50])
ax[3].set_ylim((-10, 60))
ax[4].set_yticks(np.logspace(1, 4, 4))
ax[4].set_yticklabels([f"$10^{x}$" for x in range(1, 5)])

ax[1].legend(loc="upper right", fontsize=8)
ax[2].legend(loc="upper right", fontsize=8)
ax[3].legend(loc="upper right", fontsize=8)

cbar = plt.colorbar(im, cax=ax2[-1])
cbar.set_ticks([1e3, 1e5, 1e7])
cbar.set_label("$E_i$\n$\left[\\frac{keV}{cm^2\, s\, sr\, keV}\\right]$")

for i in range(len(ax2) - 1):
    ax2[i].axis("off")

print((fgm_time[-1] - fgm_time[0]) / 60)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(oss.new_path(oss.get_path(__file__))("summary_17deg.png"), dpi=300)
plt.savefig(oss.new_path(oss.get_path(__file__))("summary_17deg.pdf"), dpi=300)
plt.show()
