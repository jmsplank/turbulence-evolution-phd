import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers.os_shortcuts import get_path, new_path
from phdhelper.helpers import override_mpl
import time

override_mpl.override()

fsm_path = new_path(get_path(__file__), "data/fsm")


def load(fpath):
    start_time = time.time()
    print(f"Loading {fpath}")
    data = np.load(fpath)
    end_time = time.time()
    print(f"File loaded in {end_time-start_time}s")
    return data


fsm_b = load(fsm_path("data.npy"))
fsm_time = load(fsm_path("time.npy"))

print(f"{fsm_time.shape = }")
splits = np.array(list(set(np.around(np.diff(fsm_time), 5))))
splits = np.diff(fsm_time)[np.diff(fsm_time) > np.min(splits) + 0.0001]
splits_index = np.nonzero(np.diff(fsm_time) == splits)
print(splits)
