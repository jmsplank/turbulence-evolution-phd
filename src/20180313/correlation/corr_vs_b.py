import numpy as np
from numpy.core.numeric import correlate
from scipy.stats import spearmanr
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.helpers import override_mpl
import matplotlib.pyplot as plt

override_mpl.override()

# fgm_path = new_path(get_path(__file__, ".."), "data/fgm")
# b = np.load(fgm_path("data.npy"))
# b_time = np.load(fgm_path("time.npy"))

fpi_path = new_path(get_path(__file__, ".."), "data/fpi")
bulkv = np.load(fpi_path("data_bulkv_i.npy"))
bulkv_time = np.load(fpi_path("time_bulkv_i.npy"))

path = new_path(get_path(__file__))
corr_lens = np.load(path("corr_lens.npy"))
corr_times = np.load(path("corr_times.npy"))

print(corr_times, corr_times.shape)


def minInd(arr, x):
    return np.argmin(np.abs(arr - x))


avgV = np.empty_like(corr_lens)
for i in range(len(corr_lens)):
    mid = corr_times[i]
    if i == 0:
        end = minInd(bulkv_time, (corr_times[0] + corr_times[1]) / 2)
        slc = slice(None, end)
    elif i == len(corr_times) - 1:
        start = minInd(bulkv_time, (corr_times[-1] + corr_times[-2]) / 2)
        slc = slice(start, None)
    else:
        start = minInd(bulkv_time, (corr_times[i] + corr_times[i - 1]) / 2)
        end = minInd(bulkv_time, (corr_times[i] + corr_times[i + 1]) / 2)
        slc = slice(start, end)
    avgV[i] = bulkv[slc, 0].mean()

print(spearmanr(avgV, corr_lens))
plt.scatter(avgV, corr_lens)
plt.show()
