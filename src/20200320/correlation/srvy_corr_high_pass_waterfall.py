import json
import warnings
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import green, red
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.physics import lengths
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.optimize import curve_fit
from scipy.signal import butter, correlate, correlation_lags, sosfilt
from tqdm import tqdm

# mpl.rcParams["agg.path.chunksize"] = 1000
override_mpl.override("|krgb")
# e = Event(__file__)

# B, B_time = e.load_fgm_srvy()

save_path = new_path(get_path(__file__))
data_path = new_path(get_path(__file__, ".."), "data")
B = np.load(data_path("fgm/data.npy"))
B_time = np.load(data_path("fgm/time.npy"))

d_i_raw = np.load(save_path("d_i.npy"))
d_i_time_raw = np.load(data_path("fpi/time_numberdensity_i.npy"))
i_vx = np.abs(np.load(data_path("fpi/data_bulkv_i.npy"))[:, 0])
i_v_time = np.load(data_path("fpi/time_bulkv_i.npy"))

with open(f"{get_path(__file__, '..')}/summary.json", "r") as file:
    summary = json.load(file)


# td = np.mean(np.diff(B_time[:1000]))  # Uneven spacing!
# td = 1.0 / 16.0
td = 1 / 128
# td = 0.06214  # Choice of [0.0625, 0.00036, 0.06214, 0.06215, 0.00037, 0.00035]


time = np.arange(
    max(B_time[0], d_i_time_raw[0], i_v_time[0]),
    min(B_time[-1], d_i_time_raw[-1], i_v_time[-1]),
    td,
)  # Evenly spaced time
data = np.empty((len(time), 3))  # Data container [x,y,z]
for i in range(3):
    f = interp1d(B_time, B[:, i])  # Linear interpolation function
    F = f(time)
    print(f"{F.shape=}")
    data[:, i] = F  # Generate evenly spaced data points
f = interp1d(d_i_time_raw, d_i_raw)
d_i = f(time)
# d_i = lengths("i", "d", number_density=nd_i, elementwise=True)
f = interp1d(i_v_time, i_vx)
vx_i = f(time)


def crop(arr, time, c):
    return arr[(time >= c[0]) & (time < c[1])]


DATA = crop(data, time, summary["crop"])
d_i = crop(d_i, time, summary["crop"])
vx_i = crop(vx_i, time, summary["crop"])
time = crop(time, time, summary["crop"])

# Tmax = 40  # Maximum scale size in seconds
# Fcrit = 1 / Tmax  # Critical frequency

TmaxA = np.logspace(np.log10(0.5), np.log10((time[-1] - time[0]) / 2 - 1), 30)
lambdas = {}
lambdas_times = {}
for Tmax in tqdm(TmaxA):
    Fcrit = 1 / Tmax
    sos = butter(
        10,
        Fcrit,
        "hp",
        fs=1.0 / td,
        output="sos",
    )  # 10th order high pass filter
    filt = DATA.copy()
    for i in range(3):
        filt[:, i] = sosfilt(sos, DATA[:, i])

    CHUNK_LEN = int(max(Tmax, (time[-1] - time[0]) / 60) // td)  # 60s in indices
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

        correlated = correlated / correlated[lags == 0]
        correlated = correlated[lags >= 0]
        lags = lags[lags >= 0]
        lags = lags * td * v_i_chunk / d_i_chunk

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

    lambdas_times[Tmax] = time[chunk_start + CHUNK_LEN // 2]
    lambdas[Tmax] = corr_lens_di


##### PLOTTING


fig = plt.figure()
# gs = fig.add_gridspec(2, 3, hspace=0, top=0.98, bottom=0.54)
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, :])
# ax2 = fig.add_subplot(gs[1])
# gs2 = fig.add_gridspec(1, 2, top=0.45, bottom=0.09, wspace=0.18)
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

ax1.plot(time, np.linalg.norm(DATA, axis=1))

X = []
Y = []
Z = []
for i in range(len(TmaxA)):
    X.extend(lambdas_times[TmaxA[i]])
    Y.extend(np.zeros(lambdas_times[TmaxA[i]].size) + TmaxA[i])
    Z.extend(lambdas[TmaxA[i]])
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
Z[Z <= 0] = np.min(Z[Z > 0])
print(f"{X.shape}|{Y.shape}|{Z.shape}")

HSLICE = 45
VSLICE = 40
XX = np.linspace(min(X), max(X), HSLICE)
YY_ = np.logspace(np.log10(min(Y)), np.log10(max(Y)), VSLICE)
XX, YY = np.meshgrid(XX, YY_)  # 2D Grid
interp = LinearNDInterpolator(list(zip(X, Y)), Z)
ZZ = interp(XX, YY)
VV = np.linspace(min(vx_i), max(vx_i), HSLICE)
VV, __ = np.meshgrid(VV, YY_)
print(f"{XX.shape}|{YY.shape}|{ZZ.shape}")

# for i in range(ZZ.shape[0]):
#     ax2.plot(
#         XX[i, :],
#         ZZ[i, :],
#         color="k",
#         alpha=np.linspace(0.01, 0.9, VSLICE)[i],
#     )
# ax2.set_yscale("log")
# ax2.set_ylabel("$\lambda_c\quad[d_i]$")
# ax2.set_xlabel("Time")
# ax1.sharex(ax2)

# secs = np.arange(time[0], time[-1], 1, int)
# mins = secs[secs % (60 * 15) == 0]
# fmt = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%H:%M")
# ax2.set_xticks(mins)
# ax2.set_xticklabels([fmt(m) for m in mins])

opacity = np.linspace(0, len(time), HSLICE + 1, dtype=int)
DNORM = np.linalg.norm(DATA, axis=1)
mean_B = np.array([DNORM[opacity[i] : opacity[i + 1]].mean() for i in range(HSLICE)])
opacity_B = mean_B - mean_B.min()
opacity_B = opacity_B / opacity_B.max()
cmap = get_cmap("custom_rgb")
for i in range(ZZ.shape[1]):
    # if opacity_B[i] == opacity_B.min() or opacity_B[i] == opacity_B.max():
    ax3.plot(
        YY[:, i],
        ZZ[:, i],
        color=cmap(opacity_B[i]),
    )
ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_xlabel("$T_{{MAX}}\quad[s]$")
ax3.set_ylabel("$\lambda_c\quad[d_i]$")

##### EXPONENTIAL
# def expFit(x, A, B, C):
#     return A * x ** B + C


# minB = np.argmin(opacity_B)
# minY = np.argmin(np.abs(YY[:, minB] - 1e0))
# maxY = np.argmin(np.abs(YY[:, minB] - 1e2))
# # print(minY, maxY)
# popt, _ = curve_fit(expFit, YY[minY:maxY, minB], ZZ[minY:maxY, minB])
# ax3.plot(YY[minY:maxY, minB], ZZ[minY:maxY, minB])
# ax3.plot(
#     YY[:, minB],
#     expFit(YY[:, minB], *popt),
#     color=red,
#     label=f"$B_{{MIN}}: {popt[0]:>4.2f} x^{{{popt[1]:>4.2f}}} + {popt[2]:>4.2f}$",
# )

# maxB = np.argmax(opacity_B)
# popt, _ = curve_fit(expFit, YY[minY:maxY, maxB], ZZ[minY:maxY, maxB])
# ax3.plot(
#     YY[:, maxB],
#     expFit(YY[:, maxB], *popt),
#     color=green,
#     label=f"$B_{{MAX}}: {popt[0]:>4.2f} x^{{{popt[1]:>4.2f}}} + {popt[2]:>4.2f}$",
# )
# ax3.legend(fontsize=7)

for i in range(VSLICE):
    ax4.scatter(
        mean_B,
        ZZ[i, :],
        color=cmap(np.linspace(0, 1, VSLICE)[i]),
        marker="x",
        alpha=0.2,
    )
ax4.scatter(
    mean_B,
    ZZ[VSLICE // 2, :],
    color=cmap(np.linspace(0, 1, VSLICE)[VSLICE // 2]),
    marker="x",
)


def line(x, m, c):
    return m * x + c


popt, _ = curve_fit(line, mean_B[1:-1], ZZ[VSLICE // 2, 1:-1])
ax4.plot(mean_B, line(mean_B, *popt), color="k")
print(popt[0])

ax4.set_yscale("log")
ax4.set_xlabel("$|B|$")
ax4.set_ylabel("$\lambda_c$")

# opacity = np.linspace(0, len(time), HSLICE + 1, dtype=int)
# mean_v = np.array([vx_i[opacity[i] : opacity[i + 1]].mean() for i in range(HSLICE)])
# opacity_v = mean_v - mean_v.min()
# opacity_v = opacity_v / opacity_v.max()
# cmap = get_cmap("custom_rgb")
for i in range(ZZ.shape[1]):
    # if opacity_v[i] == opacity_v.min() or opacity_v[i] == opacity_v.max():
    ax5.plot(
        VV[:, i],
        ZZ[:, i],
        # color=cmap(opacity_v[i]),
    )
ax5.set_yscale("log")
ax5.set_xlabel("$V_x$")
ax5.set_ylabel("$\lambda_c$")


plt.tight_layout()
# plt.subplots_adjust(hspace=0.5, wspace=0.1)
plt.show()
# plt.savefig(save_path("pcolormesh.png"), dpi=300)
