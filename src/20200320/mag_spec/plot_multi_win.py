import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
import json
from datetime import datetime as dt

override_mpl.override()
override_mpl.cmaps()

path = new_path(get_path(__file__))
fgm_path = new_path(get_path(__file__, ".."), "data/fgm")
main_path = new_path(get_path(__file__, ".."))

with open(main_path("summary.json"), "r") as file:
    summary = json.load(file)

fgm = np.load(fgm_path("data.npy"))
fgm_time = np.load(fgm_path("time.npy"))

slopes_all = np.load(path("slopes_all.npy"), allow_pickle=True)
times_all = np.load(path("times_all.npy"), allow_pickle=True)
k_extent_all = np.load(path("k_extent_all.npy"), allow_pickle=True)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 8 * (9 / 16)))

ax[1].set_yscale("log")
ax2 = ax[1].twinx()

crop = (1584732263.0706418, 1584737752.138518)

findex = lambda x, arr: np.argmin(np.abs(arr - x))
fmt = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M:%S.%f")
start = np.array(summary["burst_start_fgm"])
stop = np.array(summary["burst_stop_fgm"])

slc = (findex(crop[0], fgm_time), findex(crop[1], fgm_time))
start = start[(start >= slc[0]) & (start <= slc[1])]
stop = stop[(stop >= slc[0]) & (stop <= slc[1])]

slc = np.nonzero(
    (np.array([t[0] for t in times_all]) >= crop[0])
    & (np.array([t[-1] for t in times_all]) <= crop[1])
)
slopes_all = slopes_all[slc]
times_all = times_all[slc]
k_extent_all = k_extent_all[slc]

for i in range(len(start)):
    ax[0].plot(
        fgm_time[start[i] : stop[i]],
        fgm[start[i] : stop[i], 3],
        color="k",
    )

for i in range(len(slopes_all)):
    ax2.imshow(
        slopes_all[i].T,
        extent=(
            times_all[i][0],
            times_all[i][-1],
            k_extent_all[i][0],
            k_extent_all[i][1],
        ),
        origin="lower",
        aspect="auto",
        cmap="custom_diverging",
        vmin=-4.667,
        vmax=1.333,
    )
ax[1].set_ylim(10 ** np.array(ax2.get_ylim()))
ax2.set_yticklabels([])
ax2.set_ylabel("")

ax[1].set_xlim(fgm_time[start[0]], fgm_time[stop[-1]])

labels = np.arange(int(fgm_time[start[0]]), int(fgm_time[stop[-1]]))
labels = labels[labels % 600 == 0]
ax[-1].set_xticks(labels)
fmt = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M")
ax[-1].set_xticklabels([fmt(i) for i in labels])

ax[0].set_ylabel("$|B|$ [$nT$]")
ax[1].set_ylabel("$k$ [$km^{-1}$]")
ax[-1].set_xlabel(
    f"Time UTC {dt.strftime(dt.utcfromtimestamp(fgm_time[0]), r'%d/%m/%Y')} (HH:MM)"
)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(path("20203020_magSpec.png"), dpi=300)
plt.show()
