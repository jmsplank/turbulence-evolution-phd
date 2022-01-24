import enum
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
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner

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


# def crop(arr, time, c):
#     return arr[(time >= c[0]) & (time < c[1])]

DATA = data
# DATA = crop(data, time, summary["crop"])
# d_i = crop(d_i, time, summary["crop"])
# vx_i = crop(vx_i, time, summary["crop"])
# time = crop(time, time, summary["crop"])

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

    # print(f"{time.shape=}")
    # print(f"{filt.shape=}")
    # print(f"{d_i.shape=}")
    # print(f"{vx_i.shape=}")

    CHUNK_LEN = int(max(Tmax, (time[-1] - time[0]) / 30) // td)  # 60s in indices
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
fig, ax = plt.subplots(2, 2, gridspec_kw={"width_ratios": [97, 3]})

ax[0, 0].sharex(ax[1, 0])

##### ROW 1
ax[0, 0].plot(time, np.linalg.norm(DATA, axis=1))
ax[0, 1].axis("off")


##### ROW 2
PLOT_MIN = np.min([lambdas[l].min() for l in lambdas.keys()])
PLOT_MAX = np.max([lambdas[l].max() for l in lambdas.keys()])


spacing = np.diff(np.log10(TmaxA))[0] / 2
spacing = np.logspace(
    np.log10(TmaxA[0] - spacing), np.log10(TmaxA[-1] + spacing), TmaxA.size + 1
)
Lspacing = [np.diff(t)[0] for _, t in lambdas_times.items()]
Lspacing = [
    np.linspace(t[0] - Lspacing[i], t[-1] + Lspacing[i], t.size + 1)
    for i, (_, t) in enumerate(lambdas_times.items())
]
for i, T in enumerate(TmaxA):
    # Stack colormesh rows
    im = ax[1, 0].pcolormesh(
        Lspacing[i],
        [spacing[i], spacing[i + 1]],
        np.column_stack([lambdas[T], lambdas[T]]).T,
        norm=LogNorm(
            vmin=PLOT_MIN,
            vmax=PLOT_MAX,
        ),
    )
plt.colorbar(im, cax=ax[1, 1])


##### GEN CONTOUR DATA
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

##### CONTOURS
tri = Triangulation(X, Y)
refiner = UniformTriRefiner(tri)
tri_refi, Z_refi = refiner.refine_field(Z, subdiv=3)
im = ax[1, 0].tricontour(
    tri_refi,
    Z_refi,
    levels=np.logspace(np.log10(PLOT_MIN), np.log10(PLOT_MAX), 15),
    linewidths=np.array([2.0, 0.5, 1.0, 0.5]) / 2,
    colors="k",
)
plt.colorbar(im, ax[1, 1], label="$\lambda_c\quad[d_i]$")

ax[1, 0].set_yscale("log")
ax[1, 0].set_xlim((time[0], time[-1]))

xtick = np.arange(int(time[0]), time[-1], dtype=int)
xtick = xtick[xtick % 120 == 0]
ax[1, 0].set_xticks(xtick)
ax[1, 0].set_xticklabels([f"{dt.utcfromtimestamp(t):%H:%M}" for t in xtick])
ax[1, 0].set_xlabel(f"Time UTC {dt.utcfromtimestamp(time[0]):%d/%m/%Y}")
ax[1, 0].set_ylabel("$T_{MAX}\quad[s]$")
ax[0, 0].set_ylabel("$|B|\quad[nT]$")
plt.setp(ax[0, 0].get_xticklabels(), visible=False)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(save_path("pcolormesh.png"), dpi=300)
plt.show()
