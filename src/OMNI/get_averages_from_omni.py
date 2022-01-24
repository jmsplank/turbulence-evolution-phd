import pandas as pd
from datetime import datetime as dt
import re
from os import listdir
from os.path import dirname, join
from tabulate import tabulate

dname = dirname(__file__)
dirfiles = listdir(dname)
file_naming = re.compile(r"\d{8}_omni\.txt")
files = [join(dname, a) for a in sorted(list(filter(file_naming.match, dirfiles)))]
print(files)

for i, event in enumerate(files):
    data = pd.read_csv(
        event,
        delim_whitespace=True,
        na_values=["99999.9", "999.99", "999.9"],
    )

    data["time"] = data.apply(
        lambda row: dt.strptime(f"{int(row.HR):02d}:{int(row.MN):02d}", "%H:%M"),
        axis=1,
    )

    strt = lambda x: dt.strptime(x, "%H:%M")
    start = ["04:41", "01:40", "02:57", "19:24"][i]
    end = ["04:58", "01:57", "03:09", "20:55"][i]
    data = data.loc[(data.time >= strt(start)) & (data.time <= strt(end))]

    genstat = lambda x: f"${x.mean():.1f}\pm{x.std():.1f}$"
    tab = [
        ["filename", event],
        ["start/end", f"{start} -> {end}"],
        ["format", "v0 | mach | beta"],
        ["data", f"{genstat(data.v)} & {genstat(data.ma)} & {genstat(data.beta)}"],
    ]
    print(tabulate(tab, tablefmt="fancy_grid"))
