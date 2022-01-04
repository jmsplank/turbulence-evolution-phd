import numpy as np
from phdhelper.helpers.os_shortcuts import new_path, get_path
from os.path import join
import json
from phdhelper.helpers.CONSTANTS import mu_0, m_i


base_path = get_path(__file__, "..")
A, B, C, D = (
    join(base_path, i) for i in ["20180313", "20180316", "20200318", "20200320"]
)
print(A, B, C, D)
events = [A, B, C, D]

for event, event_label in zip(events, ["A", "B", "C", "D"]):
    # print(f"event {event.split('/')[-1]}")
    with open(join(event, "summary.json")) as file:
        data = json.load(file)
    # print(f"v0: {data['vsw']['value']:0.3f}")
    # print(f"theta_Bn: {data['theta_Bn']:0.3f}")
    # print(f"Mach: {data['Alfven_mach_number']:0.3f}")
    # print(f"Beta: {data['Plasma_beta']:0.3f}")
    # print("\n\n")
    v_alf = 2.2e1 * data["Upstream_B"] / np.sqrt(data["Proton_Density"])  # Not used
    print(
        f"{event_label} & ${data['vsw']['value']:.0f}$ & {v_alf:.0f} & ${data['theta_Bn']:.0f}$ & ${data['Alfven_mach_number']:.1f}$ & ${data['Plasma_beta']:.1f}$ \\\\"
    )
