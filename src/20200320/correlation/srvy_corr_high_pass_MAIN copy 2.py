import enum
import json
import warnings
from datetime import datetime as dt
from re import L

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from master import Event
from matplotlib.colors import LogNorm
from matplotlib.tri import TriAnalyzer, Triangulation, UniformTriRefiner
from matplotlib.patches import Rectangle
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.physics import lengths
from scipy.interpolate import interp1d
from scipy.signal import butter, correlate, correlation_lags, resample, sosfilt, welch
from scipy.spatial import Delaunay
from tqdm import tqdm

# mpl.rcParams["agg.path.chunksize"] = 1000
override_mpl.override("|krgb")
# e = Event(__file__)

# B, B_time = e.load_fgm_srvy()

save_path = new_path(get_path(__file__))
data_path = new_path(get_path(__file__, ".."), "data")
B = np.load(data_path("fsm/data.npy"))
B_time = np.load(data_path("fsm/time.npy"))

d_i_raw = np.load(save_path("d_i.npy"))
d_i_time_raw = np.load(data_path("fpi/time_numberdensity_i.npy"))
i_vx = np.abs(np.load(data_path("fpi/data_bulkv_i.npy"))[:, 0])
i_v_time = np.load(data_path("fpi/time_bulkv_i.npy"))

with open(f"{get_path(__file__, '..')}/summary.json", "r") as file:
    summary = json.load(file)

missing_data = summary["burst_stop"][:-1]
missing_data = [[B_time[m], B_time[m + 1] - B_time[m]] for m in missing_data]
missing_data = [
    m for m in missing_data if m[0] >= summary["crop"][0] and m[0] < summary["crop"][1]
]

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
    data[:, i] = F  # Generate evenly spaced data points
f = interp1d(d_i_time_raw, d_i_raw)
d_i = f(time)
# d_i = lengths("i", "d", number_density=nd_i, elementwise=True)
f = interp1d(i_v_time, i_vx)
vx_i = f(time)


def crop(arr, time, c):
    return arr[(time >= c[0]) & (time < c[1])]


# DATA = data
DATA = crop(data, time, summary["crop"])
d_i = crop(d_i, time, summary["crop"])
vx_i = crop(vx_i, time, summary["crop"])
time = crop(time, time, summary["crop"])

# Tmax = 40  # Maximum scale size in seconds
# Fcrit = 1 / Tmax  # Critical frequency

# TmaxA = np.logspace(np.log10(1), np.log10((time[-1] - time[0]) / 2 - 1), 30)
# TmaxA = TmaxA.astype(int)
# for i in range(1, len(TmaxA) - 1):
# TmaxA[i] = TmaxA[i] if TmaxA[i] > TmaxA[i - 1] else TmaxA[i - 1] + 1
# print(TmaxA)

_y = np.logspace(np.log10(2), np.log10(len(time) / 15 - 1), 30)
_y = _y.astype(int)
for i in range(1, len(_y) - 1):
    _y[i] = _y[i] if _y[i] > _y[i - 1] else _y[i - 1] + 1
TmaxA = np.array([((time[-1] - time[0]) / i) for i in _y[::-1]])

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

    # CHUNK_LEN = int(max(Tmax, (time[-1] - time[0]) / 30) // td)  # 60s in indices
    # CHUNK_LEN = int((2 * Tmax) // td)
    CHUNK_LEN = int(Tmax // td)
    chunk_start = np.arange(0, len(time) - CHUNK_LEN, CHUNK_LEN, dtype=int)
    corr_lens_di = np.empty_like(chunk_start, dtype=float)
    corr_lens_s = np.empty_like(chunk_start, dtype=float)
    for nchunk, chunk in enumerate(chunk_start):
        # ctime = time[chunk : chunk + CHUNK_LEN]
        cdata = filt[chunk : chunk + CHUNK_LEN, :]
        d_i_chunk = d_i[chunk : chunk + CHUNK_LEN].mean()
        v_i_chunk = np.nanmean(vx_i[chunk : chunk + CHUNK_LEN])

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
fig, ax = plt.subplots(
    2,
    2,
    gridspec_kw={"width_ratios": [97, 3], "height_ratios": [20, 80]},
    figsize=(7, 4),
)

# Generate false vertical axis to plot contours
ax_contour = ax[1, 0].twinx()
ax_contour.set_yticks([])  # Hide non-log axis & use ax[1,0]
ax_contour.set_yticklabels([])
ax_contour.grid(False)

# x axis manipulation
ax[0, 0].sharex(ax_contour)  # Use (top) contour axis to share with |B|
ax_contour.xaxis.set_visible(True)  # twinx x-axis invisible by default

# |B|
ax[0, 0].plot(time, np.linalg.norm(DATA, axis=1))
# ax[0,0] has no colorbar
ax[0, 1].axis("off")


# High-res colour data
PLOT_MIN = np.min([lambdas[l][lambdas[l] > 0].min() for l in lambdas.keys()]) * 100
PLOT_MAX = np.max([lambdas[l].max() for l in lambdas.keys()])

spacing = np.diff(np.log10(TmaxA))[0] / 2
spacing = np.logspace(
    np.log10(TmaxA[0]) - spacing, np.log10(TmaxA[-1]) + spacing, TmaxA.size + 1
)
Lspacing = [abs(t[1] - t[0]) for _, t in lambdas_times.items()]
Lspacing = [
    np.linspace(t[0] - Lspacing[i] / 2, t[-1] + Lspacing[i] / 2, t.size + 1)
    for i, (_, t) in enumerate(lambdas_times.items())
]

for i, T in enumerate(TmaxA):
    # Stack colormesh rows
    im = ax[1, 0].pcolormesh(
        Lspacing[i],
        # Can't plot a 1D colormesh
        [spacing[i], (spacing[i] + spacing[i + 1]) / 2, spacing[i + 1]],
        np.column_stack([lambdas[T], lambdas[T]]).T,
        norm=LogNorm(
            vmin=PLOT_MIN,
            vmax=PLOT_MAX,
        ),
    )
c_bar = plt.colorbar(im, cax=ax[1, 1])
c_bar.ax.set_ylabel("$\lambda_c\quad[d_i]$")


# contours from sampled data

# Create some regular spacings
X = lambdas_times[TmaxA[24]]  # Choose an X spacing
X2, Y2 = np.meshgrid(X, TmaxA)

# Interpolate lambdas into regular grid
lambdas_regular = np.zeros(X2.shape)
for i in range(len(TmaxA)):
    lambdas_regular[i, :] = np.interp(X, lambdas_times[TmaxA[i]], lambdas[TmaxA[i]])

# Generate triangulation on sampled grid
tri = Triangulation(X2.flatten(), np.log10(Y2).flatten())
# Refine grid to smooth out contours
UTR = UniformTriRefiner(tri)
tri_refi, lambda_refi = UTR.refine_field(np.log10(lambdas_regular).flatten(), subdiv=4)

# Contour levels
c_min, c_max = -2, 2.1
c_lev = np.arange(c_min, c_max, 0.5)
c_lev_2 = np.arange(c_min, c_max, 1 / 8)

# Plot smoothed contours
ax_contour.tricontour(  # Main solid
    tri_refi,
    lambda_refi,
    colors="k",
    levels=c_lev,
    linewidths=1,
    linestyles="solid",
    vmin=c_min,
    vmax=c_max,
)
ax_contour.tricontour(  # Secondary dashed
    tri_refi,
    lambda_refi,
    colors="k",
    levels=c_lev_2,
    linewidths=1,
    linestyles="dashed",
    alpha=0.2,
    vmin=c_min,
    vmax=c_max,
)

# Cover missing data
y0 = ax[0, 0].get_ylim()
y1 = ax_contour.get_ylim()
for m in missing_data:
    r0 = Rectangle(
        (m[0], y0[0]),  # Bottom left
        m[1],  # width (duration)
        np.diff(y0)[0],  # height
        ec="none",
        fc="white",
        zorder=2.1,
    )
    r1 = Rectangle(
        (m[0], y1[0]),
        m[1],
        np.diff(y1)[0],
        ec="none",
        fc="white",
        zorder=2.1,
    )
    ax[0, 0].add_patch(r0)  # Cover |B|
    ax_contour.add_patch(r1)  # Cover colormesh and contour

# Move axis ticks & grid above rectangles
ax[0, 0].set_axisbelow(False)
ax_contour.set_axisbelow(False)

ax[1, 0].set_yscale("log")
ax[1, 0].set_xlim((time[0], time[-1]))

xtick = np.arange(int(time[0]), time[-1], dtype=int)
xtick = xtick[xtick % 600 == 0]
ax_contour.set_xticks(xtick)
ax_contour.set_xticklabels([f"{dt.utcfromtimestamp(t):%H:%M}" for t in xtick])

ax[1, 0].set_xlabel(f"Time UTC {dt.utcfromtimestamp(time[0]):%d/%m/%Y}")
ax[1, 0].set_ylabel("$T_{max}\quad[s]$")
ax[0, 0].set_ylabel("$|B|\quad[nT]$")

plt.setp(ax[0, 0].get_xticklabels(), visible=False)

plt.tight_layout()
plt.subplots_adjust(hspace=0)

plt.savefig(save_path(f"{dt.utcfromtimestamp(time[0]):%Y%m%d}_corr.png"), dpi=300)
plt.savefig(save_path(f"{dt.utcfromtimestamp(time[0]):%Y%m%d}_corr.pdf"), dpi=300)
plt.show()
