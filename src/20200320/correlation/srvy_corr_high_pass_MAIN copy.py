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
from bisect import bisect_left as bsl
import multiprocessing as mp

# mpl.rcParams["agg.path.chunksize"] = 1000
override_mpl.override("|krgb")
# e = Event(__file__)

# B, B_time = e.load_fgm_srvy()
# def init(cross_corr, input_arr):
#     globals()["cross_corr"] = cross_corr
#     globals()["input_arr"] = input_arr


# def worker(i):
#     print(f"{mp.current_process().name} working on {i}")
#     global cross_corr
#     global input_arr
#     rolled = np.roll(input_arr, i)
#     rolled[:i] = 0
#     cross_corr[i] = np.nanmean(rolled * input_arr)
#     print(f"{mp.current_process().name} done {i}")


# def crosscorr(x):

#     cross_corr = mp.Array("d", np.empty_like(x), lock=False)
#     input_arr = x
#     mp.Pool(8, initializer=init, initargs=(cross_corr, input_arr)).map(
#         worker, range(len(x))
#     )
#     return cross_corr


def main():
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

    # thresh = 1  # seconds

    # big_diffs = np.diff(B_time) > thresh
    # index_big_diffs = np.nonzero(big_diffs)[0]

    # for index in index_big_diffs:
    #     minval = min([bsl(time, B_time[index]), len(time) - 1])
    #     maxval = min([bsl(time, B_time[index + 1]), len(time) - 1])
    #     data[minval:maxval, :] = np.nan

    def crop(arr, time, c):
        return arr[(time >= c[0]) & (time < c[1])]

    DATA = crop(data, time, summary["crop"])
    d_i = crop(d_i, time, summary["crop"])
    vx_i = crop(vx_i, time, summary["crop"])
    time = crop(time, time, summary["crop"])
    # DATA = data

    # Tmax = 40  # Maximum scale size in seconds
    # Fcrit = 1 / Tmax  # Critical frequency

    TmaxA = np.logspace(np.log10(0.5), np.log10((time[-1] - time[0]) / 4 - 1), 30)
    lambdas = {}
    lambdas_times = {}
    for Tmax in tqdm(TmaxA):
        Fcrit = 1 / Tmax
        # print(Fcrit)
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

        # CHUNK_LEN = int(max(Tmax, (time[-1] - time[0]) / 30) // td)  # 60s in indices
        CHUNK_LEN = int((2 * Tmax) // td)
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
                correlated[:, i] = correlate(cdata[:, i], cdata[:, i])
            correlated = correlated.mean(axis=1)
            lags = correlation_lags(cdata[:, i].size, cdata[:, i].size, mode="full")
            # lags = np.arange(len(cdata[:, i]))
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
    PLOT_MIN = min(lambdas)
    PLOT_MAX = 50

    spacing = np.diff(np.log10(TmaxA))[0] / 2
    spacing = np.logspace(
        np.log10(TmaxA[0] - spacing), np.log10(TmaxA[-1] + spacing), TmaxA.size + 1
    )

    Lspacing = [abs(t[1] - t[0]) for _, t in lambdas_times.items()]
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
    plt.colorbar(im, cax=ax[1, 1], label="$\lambda_c\quad[d_i]$")

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
    # tri = Triangulation(X, Y)
    # refiner = UniformTriRefiner(tri)
    # tri_refi, Z_refi = refiner.refine_field(Z, subdiv=3)
    # im = ax[1, 0].tricontour(
    #     tri_refi,
    #     Z_refi,
    #     levels=np.logspace(np.log10(PLOT_MIN), np.log10(PLOT_MAX), 15),
    #     linewidths=np.array([2.0, 0.5, 1.0, 0.5]) / 2,
    #     colors="k",
    # )
    # plt.colorbar(im, ax[1, 1])

    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlim((time[0], time[-1]))

    ticks = np.arange(int(time[0]), time[-1], dtype=int)
    ticks = ticks[np.nonzero(ticks % (10 * 60) == 0)]
    ax[-1, 0].set_xticks(ticks)
    ax[-1, 0].set_xticklabels([f"{dt.utcfromtimestamp(tick):%H:%M}" for tick in ticks])

    ax[0, 0].set_ylabel("$|B|$")
    ax[-1, 0].set_xlabel(f"Time UTC {dt.utcfromtimestamp(time[0]):%d/%m/%Y} (HH:MM)")
    ax[1, 0].set_ylabel("$T_{max}\,[s]$")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(save_path("20200320_corr.png"), dpi=300)
    plt.savefig(save_path("20200320_corr.pdf"), dpi=300)


if __name__ == "__main__":
    main()
