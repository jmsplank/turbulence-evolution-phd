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
path_20 = gen_path("20200320")

fgm = "data/fgm"
mag = join(fgm, "data.npy")
mag_time = join(fgm, "time.npy")

path = path_20

data = np.load(path(mag))[:, 3]
time = np.load(path(mag_time))

plt.plot(time, data)
plt.show()
