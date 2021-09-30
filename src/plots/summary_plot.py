import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import os_shortcuts, override_mpl
from phdhelper.helpers.CONSTANTS import R_e
from phdhelper.helpers.COLOURS import red, green, blue
from os.path import join
import json
from datetime import datetime as dt

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

fig, ax = plt.subplots(3, 3, figsize=(10, 6))

for shock, path in enumerate([path_13, path_16, path_18]):
    B = np.load(path(mag))
    B_time = np.load(path(mag_time))

    print(B_time[-1] - B_time[0])

    V_i = np.load(path(vi))
    V_i_time = np.load(path(vi_time))

    ax[0, shock].plot(B_time, B[:, 3], color="k")
    for i in range(3):
        ax[1, shock].plot(
            B_time, B[:, i], label=[f"GSE {x}" for x in ["X", "Y", "Z"]][i]
        )
        ax[2, shock].plot(
            V_i_time, V_i[:, i], label=[f"GSE {x}" for x in ["X", "Y", "Z"]][i]
        )
    ax[1, shock].legend(fontsize=8, loc="upper right")
    ax[2, shock].legend(fontsize=8, loc="lower left")

    ax[1, shock].get_shared_x_axes().join(ax[0, shock], ax[1, shock])
    ax[2, shock].get_shared_x_axes().join(ax[0, shock], ax[2, shock])

    quarters = lambda frac: B_time[int(len(B_time) * frac)]
    fmt = lambda t: dt.strftime(dt.utcfromtimestamp(t), "%H:%M:%S")
    for i in range(3):
        ax[i, shock].set_xticks([quarters(x) for x in [1 / 6, 1 / 2, 5 / 6]])
        # ax[i, shock].grid(None)

    ax[2, shock].set_xticklabels([fmt(quarters(x)) for x in [1 / 6, 1 / 2, 5 / 6]])

    ax[0, shock].set_xticklabels([])
    ax[1, shock].set_xticklabels([])

for i in range(3):
    ax[i, 1].sharey(ax[i, 0])
    ax[i, 2].sharey(ax[i, 1])

    plt.setp(ax[i, 1].get_yticklabels(), visible=False)
    plt.setp(ax[i, 2].get_yticklabels(), visible=False)

ax[0, 0].set_ylim((0, 50))
ax[1, 0].set_ylim((-50, 50))
ax[2, 0].set_ylim((-700, 300))

ax[0, 0].set_ylabel("$|B|$ [$nT$]")
ax[1, 0].set_ylabel("$B$ [$nT$]")
ax[2, 0].set_ylabel("$v_i$ [$kms^{-1}$]")

ax[2, 0].set_xlabel("13 Mar 2018")
ax[2, 1].set_xlabel("16 Mar 2018")
ax[2, 2].set_xlabel("18 Mar 2020")


plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(join(os_shortcuts.get_path(__file__), "summary_plot.pdf"), dpi=300)
plt.show()
