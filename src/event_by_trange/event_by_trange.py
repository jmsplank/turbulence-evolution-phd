import numpy as np
import matplotlib.pyplot as plt
from phdhelper.helpers import override_mpl, os_shortcuts
from phdhelper.helpers.CONSTANTS import R_e
from numpyencoder import NumpyEncoder
import json

# import pyspedas
import pybowshock as pybs

# from pytplot import data_quants

override_mpl.override()

trange = ["2018-03-13/06:11:49", "2018-03-13/07:06:26"]
in_str = "2020  80 19 30    2.85   -1.61   -0.90   386.4 13.5 -5 -3.3"
_ = lambda x: np.array(x)


def OMNI_or_MMS(which_one: int):
    if which_one == 0:
        print("OMNI")
        args = in_str.split()
        args = [float(a) for a in args]
        avgB = _(args[4:7])
        avgV = args[7]
        avgR = _(args[-3:])
    elif which_one == 1:
        print("MMS")
        # pyspedas.mms.fgm(
        #     trange=trange,
        #     probe="1",
        #     data_rate="srvy",
        #     time_clip=True,
        # )

        # b_data = data_quants["mms1_fgm_b_gse_srvy_l2"].values

        # r_data = data_quants["mms1_fgm_r_gse_srvy_l2"].values

        # avgB = b_data[:, :3].mean(axis=0)
        # avgR = r_data[:, :3].mean(axis=0) / R_e

        # pyspedas.mms.fpi(
        #     trange=trange,
        #     probe="1",
        #     data_rate="fast",
        #     time_clip=True,
        # )

        # v_data = data_quants["mms1_dis_bulkv_gse_fast"]
        # avgV = np.linalg.norm(v_data, axis=1).mean()
    else:
        raise (NotImplementedError())
    avgB = avgB / np.linalg.norm(avgB)
    return avgB, avgV, avgR


avgB, avgV, avgR = OMNI_or_MMS(0)


# print(f"{avgB=}")
# print(f"{avgR=}")
# print(f"{avgV=}")

n_sh = pybs.bs_normal_at_surf_GSE(avgR, avgV, "BS: Peredo")
# print(f"{n_sh=}")
theta_Bn = np.rad2deg(np.arccos(np.clip(np.dot(avgB, n_sh), -1, 1)))
if (theta_Bn > 90) or (theta_Bn <= 0):
    theta_Bn = np.rad2deg(np.arccos(np.clip(np.dot(-avgB, n_sh), -1, 1)))

# print(f"{theta_Bn=}")

summary = {
    "sw_b": avgB,
    "sw_v": avgV,
    "r_at_shock": avgR,
    "shock_normal": n_sh,
    "theta_bn": theta_Bn,
}
print(json.dumps(summary, cls=NumpyEncoder))
