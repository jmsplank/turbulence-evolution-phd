import numpy as np
from matplotlib import pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.helpers.COLOURS import mandarin, blue, green

override_mpl.override()
override_mpl.cmaps(name="custom_diverging")
data_path = new_path(get_path(__file__, ".."), "20200320")

slopes = np.load(data_path("mag_spec/slopes_all.npy"), allow_pickle=True)
times = np.load(data_path("mag_spec/times_all.npy"), allow_pickle=True)
k_extent = np.load(data_path("mag_spec/k_extent_all.npy"), allow_pickle=True)
B = np.load(data_path("data/fgm/data.npy"))
B_t = np.load(data_path("data/fgm/time.npy"))
print(slopes.shape)

ks = np.logspace(*k_extent[0], 32)
print(ks)
num_wins = sum([x.shape[0] for x in slopes])
the_slopes = np.empty((num_wins, 32))
the_times = np.empty(num_wins)
count = 0
for i in range(len(slopes)):
    l = slopes[i].shape[0]
    the_slopes[count : count + l, :] = slopes[i]
    the_times[count : count + l] = times[i]
    count += l


FS = [the_times[0], 1584735999]
STR = [1584735999, 1584736664]
DS = [1584736664, 1584737805]

plt.plot(ks, np.ones_like(ks) * (-5 / 3), color="k", alpha=0.8, label="$-5/3$")
for i, reg in enumerate([FS, STR, DS]):
    data = the_slopes[(the_times > reg[0]) & (the_times < reg[1]), :]
    data_m = data.mean(axis=0)
    data_s = data.std(axis=0) / np.sqrt(data.shape[0])
    plt.fill_between(
        ks,
        data_m + data_s,
        data_m - data_s,
        color=[mandarin, blue, green][i],
        lw=0,
        alpha=0.2,
        step="mid",
    )
    plt.step(
        ks,
        data_m,
        color=[mandarin, blue, green][i],
        where="mid",
        label=["Foreshock", "Transition", "Downstream"][i],
    )
plt.legend()
plt.xscale("log")

plt.tight_layout()
plt.show()
