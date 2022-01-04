import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import red
from bisect import bisect_left as bsl

override_mpl.override("|krgb")
# np.random.seed(sum([ord(x) for x in "Paul Atreides"]))

a = np.cumsum(np.abs(np.random.normal(15, 10, 50)))  # original x
b = np.linspace(a[0], a[-1], 100)  # regular x

y = np.sin(a * 10 * np.pi / max(a))  # original y

f = interp1d(a, y, kind="cubic")
z = f(b)

THRESH = 25  # seconds
big_diffs = np.diff(a) > THRESH
index_big_diffs = np.nonzero(big_diffs)[0]
print(index_big_diffs)

for index in index_big_diffs:
    # plt.subplot(2, 1, 1)
    plt.axvspan(a[index], a[index + 1], color="k", alpha=0.2)
    minval = min([bsl(b, a[index]), len(z) - 1])
    maxval = min([bsl(b, a[index + 1]), len(z) - 1])
    z[minval:maxval] = np.nan


# plt.subplot(2, 1, 1)
plt.scatter(a, y, marker="x")
# plt.subplot(2, 1, 2)
plt.plot(b, z)


plt.tight_layout()
plt.show()
