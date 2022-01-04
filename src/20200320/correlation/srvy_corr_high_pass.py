from re import L
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from master import Event
from datetime import datetime as dt
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt, welch
from tqdm import tqdm
from scipy.signal import correlate, correlation_lags
import warnings
import json
from phdhelper.physics import lengths

# mpl.rcParams["agg.path.chunksize"] = 1000
override_mpl.override("|krgb")
# e = Event(__file__)

# B, B_time = e.load_fgm_srvy()

save_path = new_path(get_path(__file__))
data_path = new_path(get_path(__file__, ".."), "data")
B = np.load(data_path("fgm/data.npy"))
B_time = np.load(data_path("fgm/time.npy"))

i_nd = np.load(save_path("d_i.npy"))
i_nd_time = np.load(data_path("fpi/time_numberdensity_i.npy"))
i_vx = np.abs(np.load(data_path("fpi/data_bulkv_i.npy"))[:, 0])
i_v_time = np.load(data_path("fpi/time_bulkv_i.npy"))

with open(f"{get_path(__file__, '..')}/summary.json", "r") as file:
    summary = json.load(file)


# td = np.mean(np.diff(B_time[:1000]))  # Uneven spacing!
# td = 1.0 / 16.0
td = 1 / 128
# td = 0.06214  # Choice of [0.0625, 0.00036, 0.06214, 0.06215, 0.00037, 0.00035]


time = np.arange(
    max(B_time[0], i_nd_time[0], i_v_time[0]),
    min(B_time[-1], i_nd_time[-1], i_v_time[-1]),
    td,
)  # Evenly spaced time
data = np.empty((len(time), 3))  # Data container [x,y,z]
for i in range(3):
    f = interp1d(B_time, B[:, i])  # Linear interpolation function
    F = f(time)
    print(f"{F.shape=}")
    data[:, i] = F  # Generate evenly spaced data points
f = interp1d(i_nd_time, i_nd)
nd_i = f(time)
d_i = lengths("i", "d", number_density=nd_i, elementwise=True)
f = interp1d(i_v_time, i_vx)
vx_i = f(time)


def crop(arr, time, c):
    return arr[(time >= c[0]) & (time < c[1])]


DATA = crop(data, time, summary["crop"])
d_i = crop(d_i, time, summary["crop"])
vx_i = crop(vx_i, time, summary["crop"])
time = crop(time, time, summary["crop"])

Tmax = 40  # Maximum scale size in seconds
Fcrit = 1 / Tmax  # Critical frequency

TmaxA = np.arange(10, 300, 25)

print(1.0 / td)
sos = butter(10, Fcrit, "hp", fs=1.0 / td, output="sos")  # 10th order high pass filter
filt = DATA.copy()
for i in range(3):
    filt[:, i] = sosfilt(sos, DATA[:, i])

DATA = np.linalg.norm(DATA, axis=1)
FILT = np.linalg.norm(filt, axis=1)

print(f"{time.shape=}")
print(f"{filt.shape=}")
print(f"{d_i.shape=}")
print(f"{vx_i.shape=}")

fig, ax = plt.subplots(1, 1, sharex=True)
f, pxx = welch(
    data[:, 0], fs=1.0 / td, window="hann", scaling="density", nperseg=len(time)
)
ax.loglog(f, np.sqrt(pxx))
f, pxx = welch(
    filt[:, 0], fs=1.0 / td, window="hann", scaling="density", nperseg=len(time)
)
ax.loglog(f, np.sqrt(pxx))
ax.axvline(Fcrit)

print(f"{f[1]=} | {f.max()=}")
print(f"{time[-1]-time[0]=}")

ax.set_ylabel("power spectral density")
ax.set_xlabel("frequency [Hz]")
plt.savefig(save_path(f"psd{Tmax}.png"), dpi=300)
plt.close()

CHUNK_LEN = int(120 // td)  # 60s in indices
chunk_start = np.arange(0, len(time) - CHUNK_LEN, CHUNK_LEN, dtype=int)
corr_lens_di = np.empty_like(chunk_start, dtype=float)
corr_lens_s = np.empty_like(chunk_start, dtype=float)
for nchunk, chunk in enumerate(chunk_start):
    # ctime = time[chunk : chunk + CHUNK_LEN]
    cdata = filt[chunk : chunk + CHUNK_LEN, :]
    d_i_chunk = d_i[chunk : chunk + CHUNK_LEN].mean()
    v_i_chunk = vx_i[chunk : chunk + CHUNK_LEN].mean()

    correlated = np.empty((cdata.shape[0] * 2 - 1, 3))
    for i in range(3):
        correlated[:, i] = correlate(cdata[:, i], cdata[:, i], mode="full")
    correlated = correlated.mean(axis=1)
    lags = correlation_lags(cdata[:, i].size, cdata[:, i].size, mode="full")
    lags = lags * td * v_i_chunk / d_i_chunk

    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]

    try:
        int_c = correlated[: np.where(correlated <= 0)[0][0]]
        int_l = lags[: np.where(correlated <= 0)[0][0]]
    except IndexError:
        warnings.warn("Correlation does not cross zero", RuntimeWarning)
        int_c = correlated
        int_l = lags

    fit = np.trapz(int_c, int_l)
    corr_lens_di[nchunk] = fit
    fit = np.trapz(int_c, int_l * d_i_chunk / v_i_chunk)
    corr_lens_s[nchunk] = fit

corr_lens_times = time[chunk_start + CHUNK_LEN // 2]

fig, ax = plt.subplots(4, 1, sharex=True)

ax[0].plot(time, DATA)
ax[1].plot(time, FILT)
ax[2].plot(corr_lens_times, corr_lens_di)
ax[3].plot(corr_lens_times, corr_lens_s)

ax[0].set_ylabel(f"$|B|$")
ax[1].set_ylabel(rf"$|B|<\frac{{1}}{{{Tmax}}}Hz$")
ax[2].set_ylabel(f"$\lambda_c\quad[d_i]$")
ax[3].set_ylabel(f"$\lambda_c\quad[s]$")

print(f"Mean correlation length: {corr_lens_di.mean():0.2f}±{corr_lens_di.std():0.2f}")

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
