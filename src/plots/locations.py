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

fgm = "data/fgm"
r_gse = join(fgm, "data_r_gse.npy")
r_gse_time = join(fgm, "time_r_gse.npy")

x_bs = np.linspace(-25, 25, 200)
y_bs = np.linspace(-25, 25, 200)

fig, ax = plt.subplots(figsize=(7, 6))
colours = [red, green, blue]
names = ["13/03/2018", "16/03/2018", "18/03/2020"]
for i, src in enumerate([path_13, path_16, path_18]):
    pos = np.load(src(r_gse))[:, :3] / R_e
    pos_mean = pos.mean(axis=0)

    with open(src("summary.json"), "r") as file:
        summary = json.load(file)

    # sigma = pybs.get_scaling(summary["nsw"]["value"], summary["vsw"]["value"])
    sigma = pybs.bs_scale_to_pos(pos_mean, pybs.model_names()[0])
    S_bs = pybs.bs_surface_on_XY_grid_GSE(
        x_bs,
        y_bs,
        summary["vsw"]["value"],
        pybs.model_names()[0],
        sigma,
        z=pos_mean[2],
    )

    plt.contour(
        x_bs,
        y_bs,
        S_bs,
        0,
        colors=colours[i],
    )

    ax.plot(pos[:, 0], pos[:, 1], color=colours[i])
    ax.scatter(pos_mean[0], pos_mean[1], color=colours[i], label=names[i])
    ax.set_xlim((25, -25))
    ax.set_ylim((-25, 25))

ax.scatter(0, 0, color="k", label="Earth")
ax.set_aspect("equal")
ax.legend()
ax.set_xlabel("GSE X [R_e]")
ax.set_ylabel("GSE Y [R_e]")
plt.tight_layout()
plt.show()
