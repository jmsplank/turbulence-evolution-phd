"""20200320 is spread over multiple burst intervals.
As such, code for this event has to be re-written.
"""
import numpy as np
from phdhelper.helpers.os_shortcuts import get_path, new_path
import json
from numpyencoder import NumpyEncoder
from datetime import datetime as dt

fsm_path = new_path(get_path(__file__, ".."), "data/fsm")
all_b = np.load(fsm_path("data.npy"))
all_b_time = np.load(fsm_path("time.npy"))

main_path = new_path(get_path(__file__, ".."))

diff = np.diff(all_b_time)
diff_round = np.round(diff, 5)
diff_unique = np.array(list(set(diff_round)))
diff_unique = diff_unique[diff_unique > diff_unique.min()]
print(diff_unique)
burst_stop = np.where(np.in1d(diff_round, diff_unique))[0]
print(burst_stop)
burst_start = burst_stop + 1

burst_start = np.insert(burst_start, 0, 0)
burst_stop = np.append(burst_stop, len(all_b_time) - 1)

with open(main_path("summary.json"), "r") as file:
    summary = json.load(file)

summary["burst_start"] = burst_start
summary["burst_stop"] = burst_stop
fmt = lambda x: dt.strftime(dt.utcfromtimestamp(x), "%Y-%m-%d/%H:%M:%S")
summary["trange"] = [fmt(all_b_time[0]), fmt(all_b_time[-1])]

with open(main_path("summary.json"), "w") as file:
    json.dump(
        summary,
        file,
        indent=4,
        sort_keys=True,
        separators=(", ", ": "),
        ensure_ascii=False,
        cls=NumpyEncoder,
    )
