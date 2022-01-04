from phdhelper.helpers import override_mpl
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
from tqdm import tqdm

override_mpl.override("|krgb")

N = 1000
x = np.linspace(0, 10 * np.pi, N)


def crosscorr(x):
    tr = np.empty_like(x)
    for i in range(len(x)):
        rolled = np.roll(x, i)
        rolled[:i] = 0
        tr[i] = np.nanmean(rolled * x)
    return tr


def yfunc(x):
    y = np.zeros_like(x)
    for _ in range(np.random.poisson(30)):
        y += np.sin(np.random.random() * 15 * x) * np.random.random()
    return y


y = yfunc(x)


def corr_w_missing(y, frac_missing, dx):
    mid = len(y) // 2
    start = int(mid - len(y) * frac_missing / 2)
    end = int(mid + len(y) * frac_missing / 2)

    Y = y.copy()
    Y[start:end] = np.nan

    corr = crosscorr(Y)
    corr /= corr[0]
    return np.trapz(corr, dx=dx)


repeats = 100

lags = np.arange(len(x)) * x[1]
missing = np.linspace(0, 1, 100)
ints = np.zeros_like(missing)
for j in tqdm(range(repeats)):
    for i, mis in enumerate(missing):
        ints[i] += corr_w_missing(y, mis, x[1])

ints /= repeats
plt.plot(missing, ints)
plt.show()
