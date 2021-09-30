import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl
from phdhelper.helpers.os_shortcuts import get_path, new_path
import pandas as pd
import seaborn as sns
from datetime import datetime as dt
import json

override_mpl.override()

path = get_path(__file__)
path = new_path(path)

use = "20200318"
if use == "20180313":
    name = "omni_min_def_vDvLZHKdDQ.lst"
elif use == "20180316":
    name = "omni_min_def_QCYqj4FTia.lst"
elif use == "20200318":
    name = "omni_min_def_dS4yGeSH9i.lst"
else:
    raise Exception("Get the date right!")

save_path = get_path(__file__, "..")
save_path = new_path(save_path, use)

with open(save_path("summary.json"), "r") as file:
    summary = json.load(file)

headers = [
    "Year",
    "Day",
    "Hour",
    "Minute",
    "BX",
    "BY",
    "BZ",
    "Speed",
    "Plasma_beta",
    "Alfven_mach_number",
    "Magnetosonic_Mach",
]

data = pd.read_csv(
    path(name),
    delim_whitespace=True,
    names=headers,
    na_values=[99999.9, 999.99, 999.9, 99.9],
)
data["Date"] = pd.to_datetime(
    data.Year * int(10 ** 7) + data.Day * int(10 ** 4) + data.Hour * 100 + data.Minute,
    format="%Y%j%H%M",
)

print(data.Date.iloc[0])
for head in headers:
    print(f"<{head}> = {np.nanmean(data[head]):06.2f}")


event_index = np.argmin(
    abs(data.dropna().Date - dt.utcfromtimestamp(summary["shock"]["timestamp"]))
)


print(data.dropna().iloc[event_index].Date)

summary = dict(
    summary,
    **dict(
        data.dropna().iloc[event_index][
            [
                "BX",
                "BY",
                "BZ",
                "Speed",
                "Plasma_beta",
                "Alfven_mach_number",
                "Magnetosonic_Mach",
            ]
        ]
    ),
)
print(summary)

with open(save_path("summary.json"), "w") as file:
    json.dump(summary, file)

# fig, ax = plt.subplots(5, 1)

# ax[0].hist(data.BX, bins=10)
# ax[0].hist(data.BY, bins=10)
# ax[0].hist(data.BZ, bins=10)

# ax[1].hist(data.Speed, bins=10)
# ax[2].hist(data.Plasma_beta, bins=10)
# ax[3].hist(data.Alfven_mach_number, bins=10)
# ax[4].hist(data.Magnetosonic_Mach, bins=10)

# plt.tight_layout()
# plt.show()
