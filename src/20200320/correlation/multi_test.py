import numpy as np
import multiprocessing as mp


def init(cross_corr, input_arr):
    globals()["cross_corr"] = cross_corr
    globals()["input_arr"] = input_arr


def worker(i):
    print(f"{mp.current_process().name} working on {i}")
    global cross_corr
    global input_arr
    rolled = np.roll(input_arr, i)
    rolled[:i] = 0
    cross_corr[i] = np.nanmean(rolled * input_arr)
    print(f"{mp.current_process().name} done {i}")


def crosscorr(x):

    cross_corr = mp.Array("d", np.empty_like(x), lock=False)
    input_arr = x
    mp.Pool(8, initializer=init, initargs=(cross_corr, input_arr)).map(
        worker, range(len(x))
    )
    return np.array(cross_corr)


def yfunc(x):
    out = np.zeros_like(x)
    for i in range(np.random.randint(10, 30)):
        out = out + (
            np.sin(np.random.random() * 10 * x + np.random.random() * 2 * np.pi)
            * np.random.random()
        )
    return out


def main():
    import matplotlib.pyplot as plt
    from phdhelper.helpers import override_mpl
    from scipy.signal import correlate, correlation_lags
    from phdhelper.helpers.os_shortcuts import get_path
    from scipy.interpolate import interp1d

    override_mpl.override("|krgb")

    x = np.linspace(0, 10 * np.pi, 1000)
    y = yfunc(x)  # Full

    y2 = y.copy()
    y2[len(y2) // 2 : (3 * len(y2)) // 4] = np.nan  # Partial

    f = interp1d(x[~np.isnan(y2)], y2[~np.isnan(y2)])
    y3 = f(x)  # Linear interpolated

    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.title("Input function")
    plt.plot(x, y, label="full")
    plt.plot(x, y2, label="with missing")
    plt.plot(x, y3, label="interpolated")
    plt.legend(loc="upper right")
    plt.subplot(3, 1, 2)
    plt.title("Autocorrelations")

    cross = crosscorr(y2)
    cross /= cross[0]
    lags = np.arange(len(y)) * x[1]

    plt.plot(lags, cross, label="New (on missing)")

    cross2 = correlate(y2, y2, mode="full")
    lags = correlation_lags(y.size, y.size, mode="full") * x[1]
    cross2 /= cross2[lags == 0]
    cross2 = cross2[lags >= 0]
    lags = lags[lags >= 0]

    plt.plot(lags, cross2, label="Scipy (on missing)")

    cross3 = correlate(y3, y3, mode="full")
    lags = correlation_lags(y.size, y.size, mode="full") * x[1]
    cross3 /= cross3[lags == 0]
    cross3 = cross3[lags >= 0]
    lags = lags[lags >= 0]

    plt.plot(lags, cross3, label="Scipy (on interpolated)")
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)

    cross4 = correlate(y, y, mode="full")
    lags = correlation_lags(y.size, y.size, mode="full") * x[1]
    cross4 /= cross4[lags == 0]
    cross4 = cross4[lags >= 0]
    lags = lags[lags >= 0]

    plt.plot(lags, cross4 - cross, label="new w/ nan")
    plt.plot(lags, cross4 - cross2, label="scipy w/ nan")
    plt.plot(lags, cross4 - cross3, label="scipy linear")
    plt.legend(loc="upper right")
    plt.title(f"Residuals")
    plt.ylabel(f"residual against full")

    name = "Comparison different methods"
    plt.suptitle(name)

    plt.tight_layout()
    plt.savefig(f"{get_path(__file__)}/{name.replace(' ', '_')}.png", dpi=300)


if __name__ == "__main__":
    main()
