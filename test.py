import numpy as np
from atexit import register


@register
def terminate():
    print("Oh HI")


numbahs = np.arange(100000)
for i, a in enumerate(numbahs):
    print(np.mean(a * numbahs[:i]))
