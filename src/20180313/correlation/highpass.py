import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from datetime import datetime as dt
from scipy.fft import fftfreq
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt, welch
from scipy.signal import correlate, correlation_lags
import json

override_mpl.override("|krgb")
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

td = 1 / 128

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
f = interp1d(i_v_time, i_vx)
vx_i = f(time)

data = np.linalg.norm(data, axis=1)

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

time = time - time[0]

ax1.plot(time, data)
ax1.set_title("$|B|$")

fcrit = 1 / 10
sos = butter(10, fcrit, "hp", fs=1.0 / td, output="sos")
ax2.plot(time, sosfilt(sos, data))
ax2.set_title("$|B|$ filter $0.1Hz$")

freq = fftfreq(len(time), td)
psd = np.abs(np.fft.fft((data - data.mean()) * np.hanning(len(time)))) ** 2
idx = np.nonzero(freq > 0)
ax3.loglog(freq[idx], psd[idx])
ax3.set_title("$PSD(|B|)$")

freq = fftfreq(len(time), td)
high_data = sosfilt(sos, data)
psd = np.abs(np.fft.fft(high_data * np.hanning(len(time)))) ** 2
idx = np.nonzero(freq > 0)
ax4.loglog(freq[idx], psd[idx])
ax4.set_title("$PSD(|B|)$ filter $0.1Hz$")

plt.tight_layout()
plt.show()
