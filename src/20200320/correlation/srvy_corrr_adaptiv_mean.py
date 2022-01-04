from re import L
import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from master import Event
from datetime import datetime as dt
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.signal import correlate, correlation_lags

override_mpl.override("|krgb")
e = Event(__file__)

B, B_time = e.load_fgm_srvy()

td = np.max(np.diff(B_time))  # Uneven spacing!
# td = 0.06214  # Choice of [0.0625, 0.00036, 0.06214, 0.06215, 0.00037, 0.00035]


time = np.arange(B_time[0], B_time[-1], td)  # Evenly spaced time
data = np.empty((len(time), 3))  # Data container [x,y,z]
for i in range(3):
    f = interp1d(B_time, B[:, i])  # Linear interpolation function
    data[:, i] = f(time)  # Generate evenly spaced data points


# DATA = np.linalg.norm(data, axis=1)
DATA = data
DATA = DATA[(time <= 1584740130) & (1584723944 < time)]
time = time[(1584723944 < time) & (time <= 1584740130)]

# MEAN_SECONDS = 500
# MEAN_LEN = int(MEAN_SECONDS / td)  # indices
# runningmean = np.empty(len(time) - MEAN_LEN, dtype=float)
# runningtime = time[MEAN_LEN // 2 : -MEAN_LEN // 2]
# for i in range(len(runningmean)):
#     runningmean[i] = DATA[i : i + MEAN_LEN].mean()


# plt.plot(time, DATA)
# plt.plot(runningtime, runningmean)
# plt.show()

CHUNK_LEN = 500
CHUNK_EXTEND = 500
spacing = td * CHUNK_LEN
print(spacing)
chunk_start = np.arange(
    CHUNK_EXTEND, len(time) - CHUNK_EXTEND * 2 - CHUNK_LEN, CHUNK_LEN, dtype=int
)
correlation_lengths = np.empty_like(chunk_start, dtype=float)

for chunk in tqdm(range(len(chunk_start))):
    ctime = time[chunk_start[chunk] : chunk_start[chunk] + CHUNK_LEN]
    cdata = DATA[chunk_start[chunk] : chunk_start[chunk] + CHUNK_LEN, :]
    cdata_expanded = DATA[
        chunk_start[chunk]
        - CHUNK_EXTEND : chunk_start[chunk]
        + CHUNK_LEN
        + CHUNK_EXTEND,
        :,
    ]

    correlated = np.empty((cdata.shape[0] * 2 - 1, 3))
    for i in range(3):
        cdata_normed = cdata[:, i] - cdata_expanded[:, i].mean()
        correlated[:, i] = correlate(cdata_normed, cdata_normed, mode="full")
    correlated = correlated.mean(axis=1)
    lags = correlation_lags(cdata_normed.size, cdata_normed.size, mode="full")
    lags = lags * td

    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]

    try:
        integrate_c = correlated[: np.where(correlated <= 0)[0][0]]
        integrate_l = lags[: np.where(correlated <= 0)[0][0]]
    except IndexError:
        integrate_c = correlated
        integrate_l = lags

    fit = np.trapz(integrate_c, integrate_l)
    correlation_lengths[chunk] = fit

chunk_times = time[chunk_start + CHUNK_LEN // 2]

MEDIAN_LEN = 100
rolling_med = np.empty(len(chunk_times) - MEDIAN_LEN)
rolling_time = np.empty(len(chunk_times) - MEDIAN_LEN)
for i in range(len(chunk_times) - MEDIAN_LEN):
    rolling_med[i] = np.median(correlation_lengths[i : MEDIAN_LEN + i])
    rolling_time[i] = chunk_times[MEDIAN_LEN // 2 + i]

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(time, np.linalg.norm(DATA, axis=1))
ax[1].plot(chunk_times, correlation_lengths)
ax[1].plot(rolling_time, rolling_med, lw=2)

ax[1].set_ylabel("Correlation length (s)")
ax[1].set_xlabel("time")

plt.show()
