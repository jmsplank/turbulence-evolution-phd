import logging
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, MultipleLocator
from phdhelper.helpers import COLOURS, override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, kurtosis, norm
from tqdm import tqdm

override_mpl.override()
override_mpl.cmaps("custom_diverging")

path = get_path(__file__)
path2 = get_path(__file__, "..")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"{path}/log.log", "w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

big_data = np.load(f"{path2}/data/fsm/data.npy")
big_time = np.load(f"{path2}/data/fsm/time.npy")

# p_max = 4
Ni = 10 ** (4 + 1)  # (Dudock de Wit et al. 2013) - Min no. elements for 4th moment
logger.info(f"Length of bins: {Ni}")
N = len(big_time) // Ni  # Number of bins
logger.info(f"Number of bins: {N}")
logger.info(f"Duration of bins (s): {big_time[Ni]-big_time[0]:0.4f}s")

max_index = len(big_time) - Ni
bin_starts = np.linspace(0, max_index, N, dtype=int)

num_samples = 100  # IDEAL Number of lags
lags_array = np.logspace(0, np.log10(Ni - 1), num_samples, dtype=int)
lags_array = np.unique(lags_array)
num_samples = len(lags_array)
logger.info(f"{num_samples = }")
logger.info(f"{lags_array = }")

kurt = np.empty(N)  # Container for scale independednt kurtosis @ window
bin_times = np.empty(N)  # Container for time @ each window
big_K = np.empty((num_samples, N))  # 2d container for lags @ each window
max_K = np.empty(N)  # container for max kurtosis @ each window

for i, bin in enumerate(tqdm(bin_starts)):  # Loop over windows
    data = big_data[bin : bin + Ni, :].mean(axis=1)  # B field
    time = big_time[bin : bin + Ni]  # time

    bin_times[i] = time[len(time) // 2]  # Get avg (center) time for window

    data_norm = data - data.mean()  # subtract mean

    # fig, ax = plt.subplots(1, 1)
    # density = gaussian_kde(data_norm)
    # ax.hist(data_norm, 100, density=True)
    # temp_x = np.linspace(min(data_norm), max(data_norm), 512)
    # pdf = norm.pdf(temp_x, 0, np.std(data_norm))
    # ax.plot(temp_x, pdf)
    # # ax.plot(temp_x, density(temp_x), label=kurtosis(data_norm, fisher=False))
    # plt.savefig(f"{path}/pdfs/{i}-{int(kurtosis(data_norm, fisher=False)*100)}.png")
    # plt.close()
    logger.info(
        f"data_norm %NAN: {np.count_nonzero(np.isnan(data_norm))/data_norm.size*100}"
    )
    kurt[i] = kurtosis(data_norm, fisher=False)  # calculate indep. kurtosis


fig, ax = plt.subplots(
    1,
    1,
    figsize=(6, 4),
    sharex=True,
    # gridspec_kw={"width_ratios": [95, 5]},
)

# ax1 = ax[0]
# ax2 = ax[1]
ax1 = ax.twinx()
ax2 = ax

# ax1.plot(big_time, np.linalg.norm(big_data, axis=1), color="k")  #  B
reduced_big = np.linspace(big_time[0], big_time[-1], 500)
f = interp1d(big_time, np.linalg.norm(big_data, axis=1))
reduced_B = f(reduced_big)
modb = ax1.fill_between(
    reduced_big,
    0,
    reduced_B,
    color="k",
    alpha=0.2,
    edgecolor="none",
)
ax1.grid(False)
f = interp1d(bin_times, kurt, bounds_error=False)
kurt = f(big_time)
kurt_hi = kurt.copy()
kurt_hi[kurt <= 3] = np.nan
kurt_lo = kurt.copy()
kurt_lo[kurt > 3] = np.nan
(heavytail,) = ax2.plot(big_time, kurt_hi, color=COLOURS.green, lw=1.5)
(lighttail,) = ax2.plot(big_time, kurt_lo, color=COLOURS.red, lw=1.5)
ax2.axhline(3, color="k")

ax1.set_ylim((0, 30))
ax2.set_ylim((0, 14))
ax2.set_zorder(ax1.get_zorder() + 1)
ax2.patch.set_visible(False)

ax1.set_ylabel("$|B|$ [nT]")
ax2.set_ylabel("Kurtosis, $\kappa$")
ax2.set_xlabel(f"UTC {dt.utcfromtimestamp(big_time[0]):%Y/%m/%d} [HH:MM]")

ax2.legend([modb, heavytail, lighttail], ["$|B|$ [nT]", "$\kappa>3$", "$\kappa\leq3$"])

ax2.xaxis.set_major_locator(MultipleLocator(120))
ax2.xaxis.set_major_locator(FixedLocator(ax2.get_xticks()))
ax2.xaxis.set_minor_locator(MultipleLocator(30))
ax2.set_xticklabels([f"{dt.utcfromtimestamp(a):%H:%M}" for a in ax2.get_xticks()])

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(f"{path}/{dt.utcfromtimestamp(big_time[0]):%Y%m%d}_kurt.png", dpi=300)
plt.savefig(f"{path}/{dt.utcfromtimestamp(big_time[0]):%Y%m%d}_kurt.pdf", dpi=300)
plt.show()

exit()
par1 = ax[1, 0].twinx()  # parasite axis (plot 2 scales on ax[1])

# scale independent
(p1,) = ax[1, 0].plot(bin_times - bin_times[0], kurt, color=COLOURS.red)
# scale dependent
(p2,) = par1.plot(bin_times - bin_times[0], max_K, color=COLOURS.green)

ax[1, 0].axhline(3, color="k")
ax[1, 0].set_ylabel("Scale independent kurtosis")
par1.set_ylabel("Max scale dep kurtosis")
par1.set_ylim((0, 100))

ax[1, 0].yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())


im = ax[2, 0].pcolor(
    bin_times - bin_times[0],
    lags_array,
    big_K,
    norm=LogNorm(vmax=50),
    # vmin=0,
    # vmax=50,
    shading="nearest",
)
ax[2, 0].set_yscale("log")

ax[0, 0].set_ylabel("$|B|$")
ax[2, 0].set_ylabel("Lag (index)")
ax[2, 0].set_xlabel("Time (s)")

ax[0, 1].axis("off")
ax[1, 1].axis("off")
fig.colorbar(im, cax=ax[2, 1])

for i in range(3):
    ax[i, 0].set_xscale("linear")
    ax[i, 0].set_xlim((0, bin_times[-1] - bin_times[0]))

ax[0, 0].get_shared_x_axes().join(*[ax[i, 0] for i in range(3)])

logger.info(f"max_K: mean: {max_K.mean()} std: {max_K.std()}")
logger.info(f"big_K: mean: {big_K.mean()} std: {big_K.std()}")
np.save(f"{path}/max_K.npy", max_K)
np.save(f"{path}/big_K.npy", big_K)
np.save(f"{path}/bin_times.npy", bin_times)
np.save(f"{path}/kurt.npy", kurt)

plt.tight_layout()
plt.savefig(f"{path}/kurtosis.png")
plt.show()
