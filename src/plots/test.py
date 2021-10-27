import os
from typing import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from phdhelper.helpers import override_mpl, os_shortcuts
from os.path import join
from matplotlib.colors import LogNorm

override_mpl.override()
override_mpl.cmaps(name="custom_linear")

root = os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."))

event = "20180313"

energyspec = "data/fpi/data_energyspectr_omni_i.npy"
energyspec_time = "data/fpi/time_energyspectr_omni_i.npy"
energyspec_bins = "data/fpi/bins_energyspectr_omni_i.npy"

espec = np.load(join(root(event), energyspec))
espec[espec == 0] = np.min(espec[espec != 0])
espec_time = np.load(join(root(event), energyspec_time))
espec_bins = np.load(join(root(event), energyspec_bins))[0, :]

fig, ax = plt.subplots(1, 1)

ax.pcolormesh(
    espec_time,
    espec_bins,
    espec.T,
    cmap="viridis",
    norm=LogNorm(),
    shading="nearest",
)
ax.set_yscale("log")
plt.show()
