import numpy as np
import json
from phdhelper.helpers import override_mpl
from phdhelper.helpers import os_shortcuts as oss
from phdhelper.helpers.COLOURS import red, green, blue, mandarin
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from datetime import datetime as dt

override_mpl.override()

save_path = oss.new_path(oss.get_path(__file__))
main_dir = oss.get_path(__file__, "..")

event_names = ["20180313", "20180316", "20200318", "20200320"]

fig, ax = plt.subplots(4, 1, figsize=(8, 5))

for i, event_name in enumerate(event_names):
    data_path = join(main_dir, event_name, "data/fgm")
    data = np.load(join(data_path, "data.npy"))[:, 3]
    time = np.load(join(data_path, "time.npy"))

    with open(join(main_dir, event_name, "summary.json"), "r") as file:
        summary = json.load(file)

    if "burst_start_fgm" in summary.keys():
        starts = summary["burst_start_fgm"]
        stops = summary["burst_stop_fgm"]
        for j in range(len(starts)):
            ax[i].plot(
                time[starts[j] : stops[j]],
                data[starts[j] : stops[j]],
                color="k",
            )
    else:
        ax[i].plot(time, data, color="k")

    if "crop" in summary.keys():
        ax[i].set_xlim(summary["crop"])
        data = data[(time >= summary["crop"][0]) & (time < summary["crop"][1])]
        time = time[(time >= summary["crop"][0]) & (time < summary["crop"][1])]

    sections = summary["sections"]["timestamp"]
    section_labels = summary["sections"]["label"]
    print(sections)
    sections = np.insert(sections, 0, time[0])
    sections = np.append(sections, time[-1])
    print(sections)
    im = []
    for j in range(len(sections) - 1):
        if i == 3:
            ax[i].axvspan(
                sections[j],
                sections[j + 1],
                color=[mandarin, blue, green][j],
                alpha=0.4,
            )
        elif i == 0:
            im.append(
                ax[i].axvspan(
                    sections[j],
                    sections[j + 1],
                    color=[red, green, blue, mandarin][5 - len(sections) :][j],
                    alpha=0.4,
                )
            )
        else:
            ax[i].axvspan(
                sections[j],
                sections[j + 1],
                color=[red, green, blue, mandarin][5 - len(sections) :][j],
                alpha=0.4,
            )
    if i == 0:
        plt.figlegend(handles=im, labels=section_labels, loc="upper right", fontsize=8)
    duration = int(time[-1] - time[0])
    num_secs = [120, 120, 120, 600][i]
    fmt = lambda x, pos: f"{dt.utcfromtimestamp(x):%H:%M}"
    ax[i].xaxis.set_major_locator(MultipleLocator(num_secs))
    ax[i].xaxis.set_major_formatter(FuncFormatter(fmt))

    ax[i].set_xlabel(f"{dt.utcfromtimestamp(time[0]):%Y/%m/%d}")

plt.tight_layout()

plt.savefig(save_path("region_timestamps.pdf"), dpi=300)
plt.savefig(save_path("region_timestamps.png"), dpi=300)
plt.show()
