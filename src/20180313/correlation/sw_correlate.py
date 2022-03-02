import json
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.physics import lengths
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags

override_mpl.override()

data_path = new_path(get_path(__file__, ".."), "data/fgm_srvy")
fpi_path = new_path(get_path(__file__, ".."), "data/fpi_fast")
save_path = new_path(get_path(__file__))

with open(fpi_path("stats.json"), "r") as file:
    stats = json.load(file)

meanv = stats["mean_v"]["value"]
numberdensity = np.load(fpi_path("data_numberdensity_i.npy"))
print(f"<v> = {meanv:0.2E} km/s | <numberdensity> = {numberdensity.mean():0.2E} /cm^3")

data = np.load(data_path("data.npy"))[:, :3]
time = np.load(data_path("time.npy"))  # Has uneven spacing!
td = np.diff(time).min()  # Get minimum time difference

adj_time = time - time[0]  # Start at 0
reg_time = (
    np.arange(adj_time[-1] // td) * td
)  # Generate new x-axis with regular spacing of td
reg_data = np.empty((len(reg_time), 3))  # Empty array to contain regular x, y, z
print(f"{len(reg_time)=:0.2E}")
for i in range(3):
    regularInterpolator = interp1d(adj_time, data[:, 0])  # Create interpolator
    reg_data[:, i] = regularInterpolator(reg_time)  # Interpolate onto regular grid

data = reg_data  # Rename
time = reg_time

data = data[(time > 2.554e4) & (time <= 5.827e4)]
time = time[(time > 2.554e4) & (time <= 5.827e4)]

td = time[1] - time[0]  # Generate new td (redundant)

print(f"Spacing = {td:0.2E}s")
gt = lambda s: dt.strftime(dt.utcfromtimestamp(s), r"%Y-%m-%d/%H:%M:%S")  # Debug
print(f"Start: {gt(time[0])} | End: {gt(time[-1])}")

ion_intertial_length = lengths(
    s="i", number_density=numberdensity, out="d"
)  # Get ion inertial length
print(f"{ion_intertial_length = :0.2E}km")

correlated = np.empty((data.shape[0] * 2 - 1, 3))  # Empty to contain correlated data
for i in range(3):
    normed_mag = data[:, i] - data[:, i].mean()  # Centre on mean
    correlated[:, i] = correlate(normed_mag, normed_mag, mode="full")  # Correlate
correlated = correlated.mean(axis=1)  # Average x, y, z
lags = correlation_lags(
    normed_mag.size, normed_mag.size, mode="full"
)  # Get lags as index
correlated = correlated[lags >= 0]  # Only want +ve lag (symmetrical)
lags = lags[lags >= 0]
correlated = correlated / correlated[0]  # Scale so correlation = 1 at lag 0

lags_s = td * lags  # Get seconds by * time difference
lags_km = lags_s * meanv  # Get distance by * ion mean v
lags_di = lags_km / ion_intertial_length  # ion inertial units by / ion inert. len.

uncorrelated = np.nonzero(np.diff(np.sign(correlated)))[0][
    0
]  # Uncorrelated at first + -> - sign change
unco = lambda u: u[uncorrelated]

corr_len = np.trapz(
    correlated[:uncorrelated], lags_di[:uncorrelated]
)  # Integrate to get correlation length

plt.figure(figsize=(6, 4))
plt.plot(time - time[0], data, color="k")
plt.xlabel("Time (s)")
plt.ylabel("$|B|$ [nT]")
plt.ylim((-10, 10))
plt.tight_layout()
plt.show()

exit()

fig, ax = plt.subplots()
ax.axvline(
    lags_s[uncorrelated],
    label=f"{unco(lags_s):0.2E} s\n{unco(lags_km):0.2E} km\n{unco(lags_di):0.2E} d_i",
)
ax.plot(lags_s, correlated, c="k")
ax.fill_between(
    lags_s[:uncorrelated],
    correlated[:uncorrelated],
    color="k",
    alpha=0.15,
    label=f"Correlation length:\n    {corr_len:0.2E} d_i\n    {corr_len*ion_intertial_length:0.2E} km",
)
ax.set_ylabel("Correlation")
ax.set_xlabel("Lag [s]")
ax.legend()

ax2 = ax.secondary_xaxis(-0.2, functions=(lambda x: x * meanv, lambda x: x / meanv))
ax3 = ax.secondary_xaxis(
    -0.4,
    functions=(
        lambda x: x * meanv / ion_intertial_length,
        lambda x: x * ion_intertial_length / meanv,
    ),
)
ax2.set_xlabel("Lag [km]")
ax3.set_xlabel(f"Lag [d_i = km / {ion_intertial_length:0.2E}]")
ax.set_xlim((lags_s[0], lags_s[-1]))

plt.tight_layout()
plt.savefig(save_path("SW_corr_len.png"))
plt.show()
