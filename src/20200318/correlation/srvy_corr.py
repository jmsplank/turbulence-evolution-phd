import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from master import Event
from datetime import datetime as dt
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.signal import correlate, correlation_lags

override_mpl.override()
e = Event(__file__)

B, B_time = e.load_fgm_srvy()

td = max(list(set(np.round(np.diff(B_time), 5))))  # Uneven spacing!
# td = 0.06214  # Choice of [0.0625, 0.00036, 0.06214, 0.06215, 0.00037, 0.00035]


time = np.arange(B_time[0], B_time[-1], td)  # Evenly spaced time
data = np.empty((len(time), 3))  # Data container [x,y,z]
for i in range(3):
    f = interp1d(B_time, B[:, i])  # Linear interpolation function
    data[:, i] = f(time)  # Generate evenly spaced data points

##################################################
# Use time & data as evenly spaced FGM_SRVY data
##################################################

##### SPLIT INTO CHUNKS ######
CHUNK_LEN = 150  # 10 points per chunk
SPACING = td * CHUNK_LEN  # ~9.3s when CHUNK_LEN==150
print(f"Each {SPACING}s window contains {SPACING/td} points")
chunk_start = np.arange(0, len(time) - CHUNK_LEN, CHUNK_LEN, dtype=int)
print(f"data contains {len(chunk_start)} windows")

# fig, ax = plt.subplots(1, 1)

correlation_lengths = np.empty_like(chunk_start, dtype=float)
for chunk in tqdm(range(len(chunk_start))):
    tim = time[chunk_start[chunk] : chunk_start[chunk] + CHUNK_LEN]
    dat = data[chunk_start[chunk] : chunk_start[chunk] + CHUNK_LEN, :]

    correlated = np.empty((dat.shape[0] * 2 - 1, 3))
    for i in range(3):
        # dat_subtracted_mean = dat[:, i] - dat[:, i].mean()
        dat_subtracted_mean = dat[:, i]
        correlated[:, i] = correlate(
            dat_subtracted_mean, dat_subtracted_mean, mode="full"
        )
    correlated = correlated.mean(axis=1)
    lags = correlation_lags(
        dat_subtracted_mean.size, dat_subtracted_mean.size, mode="full"
    )
    # lags = e.summary["Speed"] * td * lags  # Convert to length
    lags = td * lags  # Convert to seconds

    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]

    try:
        integrate_correlated = correlated[: np.where(correlated <= 0)[0][0]]
        integrate_lags = lags[: np.where(correlated <= 0)[0][0]]
    except IndexError:
        integrate_correlated = correlated
        integrate_lags = lags

    fit = np.trapz(integrate_correlated, integrate_lags)
    correlation_lengths[chunk] = fit

    # ax.plot(lags, correlated, color="k")
    # ax.fill_between(integrate_lags, 0, integrate_correlated)

    # plt.savefig(e.main_dir(f"correlation/anim/{chunk}.png"))
    # ax.clear()


chunk_times = time[chunk_start + CHUNK_LEN // 2]

MEDIAN_LEN = 100
rolling_med = np.empty(len(chunk_times) - MEDIAN_LEN)
rolling_time = np.empty(len(chunk_times) - MEDIAN_LEN)
for i in range(len(chunk_times) - MEDIAN_LEN):
    rolling_med[i] = np.median(correlation_lengths[i : MEDIAN_LEN + i])
    rolling_time[i] = chunk_times[MEDIAN_LEN // 2 + i]

print(f"{SPACING/max(correlation_lengths)=}")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 3))

ax[0].semilogy(
    time,
    np.linalg.norm(data, axis=1),
    label=rf"$\theta_{{Bn}}={e.summary['theta_Bn']}^\circ$",
)
ax[0].legend()

ax[1].plot(chunk_times, correlation_lengths, label="Correlation Lenth")
ax[1].plot(
    rolling_time,
    rolling_med,
    label=f"Rolling median over {MEDIAN_LEN*SPACING:0.2f}s",
)
ax[1].legend()
fmt = lambda t: dt.strftime(dt.utcfromtimestamp(t), "%d %H")
seconds = np.arange(time[0], time[-1] + 1, 1, dtype=int)
hours = seconds[np.nonzero(seconds % 7200 == 0)]
ax[1].set_xticks(hours)
ax[1].set_xticklabels([fmt(h) for h in hours])

ax[1].set_ylabel("Correlation length (s)")
ax[1].set_xlabel("Time (HH:MM:SS) on 18-03-2020")

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(e.main_dir(f"correlation/srvy_{CHUNK_LEN}.png"))
