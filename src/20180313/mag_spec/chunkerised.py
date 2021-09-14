import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from scipy.optimize import curve_fit
from tqdm import tqdm

override_mpl.override()
override_mpl.cmaps("custom_diverging")

save_path = new_path(get_path(__file__))
data_path = new_path(get_path(__file__, ".."), "data")

# Load FSM data
all_B = np.load(data_path("fsm/data.npy"))
all_time = np.load(data_path("fsm/time.npy"))
td = all_time[1] = all_time[0]

# No. Windows
N = 100
# No. Gradients to calculate per window
GRADS = 17

bins_start = np.linspace(0, len(all_time), N, endpoint=False, dtype=int)
# print(f"{bins_start[-1]=} | {len(all_time)=} | {len(all_time)-bins_start[-1]}")
bin_length = np.diff(bins_start)[0]

all_grads = []
all_grads_freq = []
all_times_mid = []

for time_bin, bin in enumerate(tqdm(bins_start)):
    data = all_B[bin : bin + bin_length, :]
    time = all_time[bin : bin + bin_length]

    components = {}
    for component in range(3):
        B = data[:, component] * 1e-09
        B -= B.mean()
        Hann = np.hanning(len(B)) * B
        Y_component = np.fft.fft(Hann)
        freq = np.fft.fftfreq(len(B), td)
        components[component] = (np.power(np.abs(Y_component), 2) * 1e9 * td)[freq > 0]
    y = np.sum([components[i] for i in range(3)], axis=0)
    freq = freq[freq > 0]

    grads_start = np.logspace(
        np.log10(freq.min()), np.log10(freq.max()), GRADS, endpoint=False
    )
    grads_start = np.array([np.argmin(np.abs(freq - g)) for g in grads_start])

    grads = []
    grads_freq = []
    for grad in range(len(grads_start) - 1):
        grad_y = y[grads_start[grad] : grads_start[grad + 1] + 1]
        grad_freq = freq[grads_start[grad] : grads_start[grad + 1] + 1]

        g, c = np.polyfit(np.log(grad_freq), np.log(grad_y), 1)
        grads.append(g)
        grads_freq.append((grad_freq[-1] + grad_freq[0]) / 2)

        if time_bin == 0:

            def func(x, A, B):
                return np.exp(A * np.log(x) + B)

            plt.plot(
                [grad_freq[0], grad_freq[-1]],
                np.array([func(grad_freq[0], g, c), func(grad_freq[-1], g, c)]),
                c="k",
                zorder=2,
                lw=1.5,
            )
    if time_bin == 0:
        plt.loglog(freq, y, zorder=1)
        plt.savefig(save_path("chunkspectrum.png"), dpi=300)
        plt.close()

    all_grads.append(grads)
    all_grads_freq.append(grads_freq)
    all_times_mid.append((time[-1] + time[0]) / 2)

# print(all_grads)
# all_grads = np.array(all_grads).flatten()
all_times_mid = np.array(all_times_mid).flatten()
all_grads_freq = np.log10(np.array(grads_freq))
all_grads_freq, all_times_mid = np.meshgrid(all_grads_freq, all_times_mid)
# all_times_mid = all_times_mid.flatten()
# all_grads_freq = all_grads_freq.flatten()

plt.figure(figsize=(8, 4))
print(f"{all_times_mid.shape=} | {all_grads_freq.shape}")
# , vmin=all_grads.min(), vmax=all_grads.max()
plt.contourf(
    all_times_mid - all_time[0],
    all_grads_freq,
    all_grads,
    levels=64,
    # norm=colors.SymLogNorm(linthresh=0.03, base=10),
    norm=colors.CenteredNorm(vcenter=-5.0 / 3, halfrange=4.333333),
    cmap="custom_diverging",
)
plt.xlabel("time (s)")
plt.ylabel("log(frequency)")

# plt.scatter(all_times_mid, all_grads_freq)

plt.colorbar()
plt.tight_layout()
plt.subplots_adjust(wspace=0.02)
plt.savefig(save_path("chunkerised.png"))
# plt.show()
