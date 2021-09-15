import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.core import numeric
import seaborn as sns
from phdhelper.helpers import os_shortcuts, override_mpl
from phdhelper.helpers.CONSTANTS import mu_0, m_i, R_e
import pybowshock as pybs

override_mpl.override()

fgm_path = os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."), "data/fgm")
save_path = os_shortcuts.new_path(os_shortcuts.get_path(__file__))
summary_path = f"{os_shortcuts.get_path(__file__, '..')}/summary.json"

with open(summary_path, "r") as file:
    summary = json.load(file)
SHOCK = summary["shock"]["timestamp"]

ALL_B = np.load(fgm_path("data.npy"))[:, :3]
ALL_B_TIME = np.load(fgm_path("time.npy"))

shock_index = np.argmin(abs(ALL_B_TIME - SHOCK))

all_B_d = ALL_B[:shock_index, :]  # Downstream B
all_B_u = ALL_B[shock_index:, :]  # Upstream B

# length = min(len(all_B_d), len(all_B_u))

###################################################
# Calculate theta_Bn by averaging an ensemble of results calculated using
# random B field measurements either side of the shock
###################################################
length = 1000
thetas = np.empty(length)
shock_normals = np.empty((length, 3))
for i in range(length):
    B_d_index = np.random.randint(0, length)
    B_d = all_B_d[B_d_index, :]
    all_B_d = np.delete(all_B_d, B_d_index, axis=0)

    B_u_index = np.random.randint(0, length)
    B_u = all_B_u[-B_u_index, :]
    all_B_u = np.delete(all_B_u, -B_u_index, axis=0)

    B_delta = B_d - B_u

    cross_d_u = np.cross(B_d, B_u)
    cross_du_delta = np.cross(cross_d_u, B_delta)

    shock_normal = cross_du_delta / np.linalg.norm(cross_du_delta)
    shock_normals[i, :] = shock_normal

    theta_Bn = np.arccos(
        np.clip(np.dot(B_u / np.linalg.norm(B_u), shock_normal), -1, 1)
    )
    thetas[i] = np.rad2deg(theta_Bn)

print(f"Theta B_n = {thetas.mean():04.1f} +/- {thetas.std():04.1f}")
shock_normal = shock_normals.mean(axis=0)

###################################################
# Shock velocity using mass flux algorithm
###################################################
fpi_path = os_shortcuts.new_path(os_shortcuts.get_path(__file__, ".."), "data/fpi")
density = (
    np.load(fpi_path("data_numberdensity_i.npy")) * 1e6
)  # Convert to m^-3 from cm^-3
density_time = np.load(fpi_path("time_numberdensity_i.npy"))
density_shock_index = np.argmin(abs(density_time - SHOCK))

bulkv = np.load(fpi_path("data_bulkv_e.npy"))
bulkv_time = np.load(fpi_path("time_bulkv_e.npy"))
bulkv_shock_index = np.argmin(abs(bulkv_time - SHOCK))

density_d = density[:density_shock_index].mean()
density_u = density[density_shock_index:].mean()

bulkv_d = bulkv[:bulkv_shock_index, :].mean(axis=0)
bulkv_u = bulkv[bulkv_shock_index:, :].mean(axis=0)

density_v_d = density_d * bulkv_d
density_v_u = density_u * bulkv_u

delta_density = density_d - density_u
delta_density_v = density_v_d - density_v_u

shock_speed = np.dot((delta_density_v / delta_density), shock_normal)
print(f"{shock_speed = :04.1f} km/s")

################################################
# Mach number
################################################

# Transform to stationary shock frame
bulkv_u_sh = bulkv_u - (shock_speed * shock_normal)
numerator = np.linalg.norm(np.dot(bulkv_u_sh, shock_normal))
denominator = (
    np.linalg.norm(all_B_u.mean(axis=0) * 1e-9) / np.sqrt(density_u * m_i * mu_0)
) / 1000

mach_number = numerator / denominator
print(f"{mach_number = :04.1f}")

# sns.histplot(thetas, bins=16)
# plt.title(f"Theta B_n = {thetas.mean():04.1f} +/- {thetas.std():04.1f}")
# plt.tight_layout()
# plt.savefig(save_path("parameters.png"))
# plt.show()

##################################################
# Comparison with model
##################################################
r = np.load(fgm_path("data_r_gse.npy"))[:, :3]
r_time = np.load(fgm_path("time_r_gse.npy"))

r = (r / R_e).mean(axis=0)
vsw = np.linalg.norm(bulkv_u)

avg_B_u = all_B_u.mean(axis=0)

model_n_sh = []
for model in pybs.model_names():
    if "BS:" in model:
        n_sh = pybs.bs_normal_at_surf_GSE(r, vsw, model)
        model_n_sh.append(n_sh)
        theta_Bn = np.rad2deg(
            np.arccos(np.clip(np.dot(avg_B_u / np.linalg.norm(avg_B_u), n_sh), -1, 1))
        )
        if theta_Bn > 90:
            theta_Bn = np.rad2deg(
                np.arccos(
                    np.clip(np.dot(-avg_B_u / np.linalg.norm(avg_B_u), n_sh), -1, 1)
                )
            )
        print(f"{model} -> {theta_Bn:02.5f}")


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
    # ptsnew[:, 4] = np.arctan2(
    #     np.sqrt(xy), xyz[:, 2]
    # )  # for elevation angle defined from Z-axis down
    ptsnew[:, 4] = np.arctan2(
        xyz[:, 2], np.sqrt(xy)
    )  # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.rad2deg(ptsnew[:, -2:])


spherical_model = appendSpherical_np(np.array(model_n_sh))
spherical_coplanar = appendSpherical_np(shock_normals)
spherical_model = np.abs(spherical_model)
mean_spherical_coplanar = spherical_coplanar.mean(axis=0)
std_spherical_coplanar = spherical_coplanar.std(axis=0)

plt.scatter(spherical_model[:, 0], spherical_model[:, 1])
plt.scatter(mean_spherical_coplanar[0], mean_spherical_coplanar[1])
plt.errorbar(
    mean_spherical_coplanar[0],
    mean_spherical_coplanar[1],
    std_spherical_coplanar[1],
    std_spherical_coplanar[0],
)
plt.show()
