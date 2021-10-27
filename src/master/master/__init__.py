"""Master functions.
"""
from typing import List
from phdhelper.helpers.os_shortcuts import get_path, new_path
import json
from pprint import pformat
import numpy as np
from os.path import join, basename


class Event:
    def __init__(self, file):
        self.main_dir = new_path(get_path(file, ".."))
        self.name = basename(self.main_dir("")[:-1])
        with open(self.main_dir("summary.json"), "r") as file:
            self.summary = json.load(file)

    def load_fgm(self):
        """Returns FGM data for event.
        OUT:
            FGM_B: [n, 4]
            FGM_TIME: [n]
        """
        fgm_dir = "data/fgm"
        data_str = self.main_dir(join(fgm_dir, "data.npy"))
        time_str = self.main_dir(join(fgm_dir, "time.npy"))
        return np.load(data_str), np.load(time_str)

    def load_fgm_srvy(self):
        """Returns the srvy data from FGM.
        OUT:
            FGM_B_SRVY: [n, 4]
            FGM_B_SRVY_TIME: [n]
        """
        fpath = "data/fgm_srvy"
        data_str = self.main_dir(join(fpath, "data.npy"))
        time_str = self.main_dir(join(fpath, "time.npy"))
        return np.load(data_str), np.load(time_str)

    def __repr__(self) -> str:
        return (
            f"Event {self.name}\nLocated at {self.main_dir('')}\nSummary:\n"
            + pformat(self.summary)
        )
