from os import wait
import numpy as np
from phdhelper.helpers.os_shortcuts import new_path, get_path
from phdhelper.helpers.COLOURS import red
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
import json
import itertools
from scipy.optimize import curve_fit


override_mpl.override()

data_path = new_path(get_path(__file__, ".."))

with open(data_path("data/fpi/stats.json")) as file:
    stats = json.load(file)

B = np.load(data_path("data/fsm/data.npy"))
B_time = np.load(data_path("data/fsm/time.npy"))
B_time = B_time - B_time[0]

lag_seconds = 2.27089484500741  # from correlation_length.py
# lag_seconds = B_time[1]

lag = np.argmin(np.abs(B_time - lag_seconds))
print(lag)
delta_B = (np.roll(B, lag) - B)[lag:]  # Add lag & exclude wrapped points
delta_B = np.linalg.norm(delta_B, axis=np.argmin(delta_B.shape))
print(f"{delta_B.shape=} {B.shape=}")
delta_T = B_time[lag:]

time = delta_T / lag_seconds

split = np.argmin(abs(100 - time))

square_delta_B = np.power(delta_B, 2)
root_mean_square = np.sqrt(square_delta_B.mean())

PVI = delta_B / root_mean_square
print(root_mean_square)

# Crop PVI by along time 280 = overshoot >300 = SW
PVI = PVI[time > 300]
time = time[time > 300]

threshold = np.mean(PVI) + np.std(PVI)

waiting = PVI <= threshold
waiting = np.sort(
    np.array(
        [sum(1 for _ in group) for key, group in itertools.groupby(waiting) if key]
    )
    * B_time[1]
)


fig, ax = plt.subplots(2, 1, figsize=(8, 6))

ax[0].plot(time, PVI, label="s")
ax[0].set_xlabel("Time / $t_c$")
ax[0].set_ylabel(rf"$PVI(s= x, y, z;\tau={lag_seconds:.4f})$")
ax[0].axhline(threshold, color="k", ls="--", label="threshold")
ax[0].legend(loc="upper right")


def linfit(x, a, b):
    return a + x * b


bin_edges = np.logspace(np.log10(waiting.min()), np.log10(waiting.max()), 15)
bins, _ = np.histogram(waiting, bins=bin_edges, density=True)
bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
bins[bins == 0] = np.nan

print(bins)
if sum(np.isnan(bins)) > 0:
    mask = ~np.isnan(bins)
    bins = bins[mask]
    bin_mids = bin_mids[mask]
    print(bins)

ax[1].loglog(
    bin_mids,
    bins,
    marker="o",
    mfc="white",
    label="s",
)

# Linear Fit

popt, pcov = curve_fit(linfit, np.log10(bin_mids), np.log10(bins))
yfit = np.power(10, linfit(np.log10(bin_mids), *popt))
ax[1].loglog(
    bin_mids,
    yfit,
    color="k",
    ls="--",
    label=f"Slope $={popt[1]:0.3f}$",
)

ax[1].legend()


ax[1].set_ylabel("Probability")
# ax[1].get_shared_y_axes().join(ax[1], ax[1])
ax[1].set_xlabel("Waiting time / $t_c$")

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.savefig(f"{get_path(__file__)}/PVI.png")
plt.show()
