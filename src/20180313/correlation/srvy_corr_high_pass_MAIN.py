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
from scipy.signal import correlate, correlation_lags, resample
from scipy.spatial import Delaunay
import warnings
import json
from phdhelper.physics import lengths
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation, UniformTriRefiner, TriAnalyzer
import warnings

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
print(TmaxA[0], TmaxA[-1])

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

ax[0, 0].sharex(ax[1, 0])

##### ROW 1
ax[0, 0].plot(time, np.linalg.norm(DATA, axis=1))
ax[0, 1].axis("off")


##### ROW 2
PLOT_MIN = np.min([lambdas[l][lambdas[l] > 0].min() for l in lambdas.keys()])
PLOT_MAX = np.max([lambdas[l].max() for l in lambdas.keys()])
print(f"{PLOT_MIN=} {PLOT_MAX=}")

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
        [spacing[i], (spacing[i] + spacing[i + 1]) / 2, spacing[i + 1]],
        np.column_stack([lambdas[T], lambdas[T]]).T,
        norm=LogNorm(
            vmin=PLOT_MIN,
            vmax=PLOT_MAX,
        ),
    )
c_bar = plt.colorbar(im, cax=ax[1, 1])
c_bar.ax.set_ylabel("$\lambda_c\quad[d_i]$")


##### GEN CONTOUR DATA
X = []
Y = []
Z = []

print([len(lambdas_times[l]) for l in list(lambdas_times.keys())])

max_bins = 30
for i in range(len(TmaxA)):
    tmax = TmaxA[i]
    lt = lambdas_times[TmaxA[i]]
    amount = lt.size
    # Z_data = lambdas[tmax]
    Z_data = np.random.normal(10, 1, amount)
    # print(Z_data.shape)
    if amount <= max_bins:
        X.extend(lt)
        Y.extend(np.zeros(amount) + tmax)
        Z.extend(Z_data)
    else:
        newX = np.linspace(lt[0], lt[-1], max_bins)
        X.extend(newX)
        Y.extend(np.zeros(max_bins) + tmax)
        bins_len = int(amount // max_bins)
        if Z_data.size % max_bins != 0:
            Z_mini = np.pad(
                Z_data.astype(float),
                (0, max_bins - Z_data.size % max_bins),
                mode="constant",
                constant_values=np.NaN,
            ).reshape(max_bins, -1)
            # print(f"if {Z_mini.shape}")
        else:
            Z_mini = Z_data.reshape(-1, bins_len)
            # print(f"else {Z_mini.shape}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Z_mean = np.nanmean(Z_mini, axis=1)
        Z.extend(Z_mean)

# np.save("/Users/jamesplank/Downloads/temp/lambdas.npy", lambdas, allow_pickle=True)
# np.save(
#     "/Users/jamesplank/Downloads/temp/lambdas_times.npy",
#     lambdas_times,
#     allow_pickle=True,
# )
# np.save("/Users/jamesplank/Downloads/temp/tmaxs.npy", TmaxA, allow_pickle=True)

##### CONTOURS

# Create some regular spacings
X = lambdas_times[TmaxA[20]]  # Choose an X spacing
X2, Y2 = np.meshgrid(X, TmaxA)

# Interpolate lambdas into regular grid
lambdas_regular = np.zeros(X2.shape)
for i in range(len(TmaxA)):
    lambdas_regular[i, :] = np.interp(X, lambdas_times[TmaxA[i]], lambdas[TmaxA[i]])

# Correction for log(lambda<=0)
# for T in TmaxA:
#     lambdas[T][lambdas[T] <= 0] = min(lambdas[T][lambdas[T] > 0])

# Generate triangulation on sampled grid
tri = Triangulation(X2.flatten(), np.log10(Y2).flatten())
# Refine grid to smooth out contours
UTR = UniformTriRefiner(tri)
tri_refi, lambda_refi = UTR.refine_field(np.log10(lambdas_regular).flatten(), subdiv=4)

# Contour levels
c_min, c_max = -2, 2.1
c_lev = np.arange(c_min, c_max, 0.5)
c_lev_2 = np.arange(c_min, c_max, 1 / 8)

# Generate false vertical axis to plot contours
ax_contour = ax[1, 0].twinx()
ax_contour.set_yticks([])
ax_contour.set_yticklabels([])

# Plot smoothed contours
ax_contour.tricontour(
    tri_refi,
    lambda_refi,
    colors="k",
    levels=c_lev,
    linewidths=1,
    linestyles="solid",
    vmin=c_min,
    vmax=c_max,
)
ax_contour.tricontour(
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
plt.savefig(save_path(f"{dt.utcfromtimestamp(time[0]):%Y%m%d}_corr.png"), dpi=300)
plt.savefig(save_path(f"{dt.utcfromtimestamp(time[0]):%Y%m%d}_corr.pdf"), dpi=300)
plt.show()
