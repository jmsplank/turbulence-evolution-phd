import numpy as np
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.helpers import override_mpl
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
from tqdm import tqdm
from bisect import bisect_left as bsl

override_mpl.override("|krgb")
data_path = new_path(get_path(__file__, ".."), "data/fgm")
B = np.load(data_path("data.npy"))
time = np.load(data_path("time.npy"))

with open(f"{get_path(__file__, '..')}/summary.json") as file:
    summary = json.load(file)


def crop(arr, t):
    return arr[np.nonzero((t >= summary["crop"][0]) & (t <= summary["crop"][1]))]


B = crop(B, time)
time = crop(time, time)
time_in = np.linspace(time[0], time[-1], int((time[-1] - time[0]) * 128))

f = interp1d(time, B, axis=0)
B = f(time_in)

thresh = 1  # seconds

big_diffs = np.diff(time) > thresh
index_big_diffs = np.nonzero(big_diffs)[0]

for index in index_big_diffs:
    minval = min([bsl(time_in, time[index]), len(time_in) - 1])
    maxval = min([bsl(time_in, time[index + 1]), len(time_in) - 1])
    B[minval:maxval, :] = np.nan

plt.scatter(time_in, B[:, 3])
plt.show()
