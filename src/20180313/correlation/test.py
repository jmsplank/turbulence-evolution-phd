from phdhelper.helpers import override_mpl
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt

override_mpl.override("|krgb")

N = 1000
x = np.linspace(0, 10 * np.pi, N)


def yfunc(x):
    return np.sin(x) + 0.6 * np.sin(2 * x) + 0.4 * np.sin(3 * x) + 0.2 * np.sin(10 * x)


y = yfunc(x)

y[300:550] = np.nan


def crosscorr(x):
    tr = np.empty_like(x)
    for i in range(len(x)):
        rolled = np.roll(x, i)
        rolled[:i] = 0
        tr[i] = np.nanmean(rolled * x)
    return tr


def fill_linear(x, arr):
    nans = np.nonzero(np.isfinite(arr) == False)[0]
    edges = [nans[0] - 1, nans[-1] + 1]

    m = (arr[edges[1]] - arr[edges[0]]) / (x[edges[1]] - x[edges[0]])
    c = arr[edges[0]] - m * x[edges[0]]

    def line(x, m, c):
        return m * x + c

    arr[nans] = line(x[nans], m, c)
    return arr


yy = fill_linear(x, y.copy())

fig, ax = plt.subplots(2, 1)

ax[0].plot(x, yy, label="Linear")
ax[0].plot(x, yfunc(x) + 0.5, label="$\sin(x)+0.5$")


corr = correlate(yy, yy)
lags = correlation_lags(yy.size, yy.size)
lags = lags * x[1]

ax[1].plot(lags, corr / max(corr), label="Linear")

corr = correlate(yfunc(x), yfunc(x))
ax[1].plot(lags, corr / max(corr), label="Reference")

# y = y.astype(np.complex64)
corr = crosscorr(y)
corr /= max(corr)
lags = np.arange(len(corr)) * x[1]
ax[1].plot(lags, corr, label="Ignored")

plt.figlegend()
plt.tight_layout()
plt.show()
