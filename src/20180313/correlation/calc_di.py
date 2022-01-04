from phdhelper.helpers.os_shortcuts import new_path, get_path
import numpy as np
from phdhelper.helpers import override_mpl
from phdhelper.physics import lengths
import matplotlib.pyplot as plt
import json
from typing import List

override_mpl.override("|krgb")

data_path = new_path(get_path(__file__, ".."), "data")

with open(f"{get_path(__file__, '..')}/summary.json", "r") as file:
    summary = json.load(file)

B = np.load(data_path("fgm/data.npy"))
B_time = np.load(data_path("fgm/time.npy"))

n_i = np.load(data_path("fpi/data_numberdensity_i.npy"))
n_i_time = np.load(data_path("fpi/time_numberdensity_i.npy"))

d_i = np.empty_like(n_i)
for i in range(n_i_time.size):
    d_i[i] = lengths("i", "d", number_density=n_i[i])

nan_index = np.nonzero(np.isfinite(d_i) == False)[0]
print(d_i[nan_index])
d_i[nan_index] = (d_i[nan_index - 1] + d_i[nan_index + 1]) / 2
print(d_i[nan_index])

plt.plot(d_i)
plt.show()
np.save(f"{get_path(__file__)}/d_i.npy", d_i)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# bsta = summary["burst_start_fgm"]
# bsto = summary["burst_stop_fgm"]
# for i in range(len(bsta)):
#     ax1.plot(B_time[bsta[i] : bsto[i]], B[bsta[i] : bsto[i], 3], "k")

# ax2.plot(n_i_time, n_i)
# ax3.plot(n_i_time, d_i)

# ax1.set_ylabel("$|B|$")
# ax2.set_ylabel("$n_i$ [$cm^{-1}$]")
# ax3.set_ylabel("$d_i$ [km]")
# plt.tight_layout()
# plt.subplots_adjust(hspace=0)
# # plt.savefig(f"{get_path(__file__)}/d_i.png", dpi=300)
# plt.show()
