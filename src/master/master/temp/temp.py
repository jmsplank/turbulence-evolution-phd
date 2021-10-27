import master
from numpy import bitwise_and
from phdhelper.helpers import override_mpl
import matplotlib.pyplot as plt

override_mpl.override()

e = master.Event("20180313")

b, b_time = e.load_fgm()
plt.plot(b_time, b)
plt.show()
