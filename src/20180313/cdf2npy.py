import os
import pyspedas
from pytplot import data_quants
import numpy as np
import logging as log
import matplotlib.pyplot as plt
from typing import Literal
from datetime import datetime as dt
from datetime import timedelta
import json
from phdhelper.helpers.os_shortcuts import get_path

path = os.path.dirname(os.path.realpath(__file__))

log.basicConfig(
    filename=f"{path}/cdf2npy.log",
    level=log.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


class Product:
    NUMBERDENSITY = "numberdensity"
    TEMPPERP = "tempperp"
    TEMPPARA = "temppara"
    BULKV = "bulkv"
    R_GSE = "r_gse"


class Instrument:
    FPI = "fpi"
    FSM = "fsm"
    FGM = "fgm"


def download_data(
    trange,
    INSTRUMENT,
    SPECIES="",
    PRODUCT="",
    data_rate: Literal["brst", "aftr"] = "brst",
    added_time=0,
):
    dir_name = f"{path}/data/{INSTRUMENT}"
    log.info(f"dir_name: {dir_name}")

    probe = "1"

    if data_rate == "aftr":
        extended_trange = [
            trange[1],
            dt.strftime(
                dt.strptime(trange[1], r"%Y-%m-%d/%H:%M:%S")
                + timedelta(hours=added_time),
                r"%Y-%m-%d/%H:%M:%S",
            ),
        ]

    if INSTRUMENT == Instrument.FGM:
        if data_rate == "aftr":
            pyspedas.mms.fgm(
                trange=extended_trange,
                probe=probe,
                data_rate="srvy",
                level="l2",
                time_clip=True,
            )
            data = data_quants["mms1_fgm_b_gse_srvy_l2"].values
            time = data_quants["mms1_fgm_b_gse_srvy_l2"].coords["time"].values
            dir_name += f"_srvy"
        else:
            pyspedas.mms.fgm(
                trange=trange,
                probe=probe,
                data_rate=data_rate,
                level="l2",
            )
            if PRODUCT == Product.R_GSE:
                data = data_quants["mms1_fgm_r_gse_brst_l2"].values
                time = data_quants["mms1_fgm_r_gse_brst_l2"].coords["time"].values
            else:
                data = data_quants["mms1_fgm_b_gse_brst_l2"].values
                time = data_quants["mms1_fgm_b_gse_brst_l2"].coords["time"].values

    elif INSTRUMENT == Instrument.FSM:
        pyspedas.mms.fsm(
            trange=trange,
            probe=probe,
            data_rate=data_rate,
            level="l3",
        )
        data = data_quants["mms1_fsm_b_gse_brst_l3"].values
        time = data_quants["mms1_fsm_b_gse_brst_l3"].coords["time"].values
    elif INSTRUMENT == Instrument.FPI:
        if data_rate == "aftr":
            pyspedas.mms.fpi(
                trange=extended_trange,
                probe=probe,
                data_rate="fast",
                level="l2",
                time_clip=True,
            )
            if PRODUCT == Product.BULKV:
                data = data_quants[f"mms1_d{SPECIES}s_bulkv_gse_fast"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_bulkv_gse_fast"]
                    .coords["time"]
                    .values
                )
            if PRODUCT == Product.NUMBERDENSITY:
                data = data_quants[f"mms1_d{SPECIES}s_numberdensity_fast"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_numberdensity_fast"]
                    .coords["time"]
                    .values
                )
            dir_name += "_fast"
        else:
            pyspedas.mms.fpi(
                trange=trange,
                probe=probe,
                data_rate=data_rate,
                level="l2",
            )
            if PRODUCT == Product.NUMBERDENSITY:
                data = data_quants[f"mms1_d{SPECIES}s_numberdensity_brst"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_numberdensity_brst"]
                    .coords["time"]
                    .values
                )
            elif PRODUCT == Product.TEMPPERP:
                data = data_quants[f"mms1_d{SPECIES}s_tempperp_brst"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_tempperp_brst"].coords["time"].values
                )
            elif PRODUCT == Product.TEMPPARA:
                data = data_quants[f"mms1_d{SPECIES}s_temppara_brst"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_temppara_brst"].coords["time"].values
                )
            elif PRODUCT == Product.BULKV:
                data = data_quants[f"mms1_d{SPECIES}s_bulkv_gse_brst"].values
                time = (
                    data_quants[f"mms1_d{SPECIES}s_bulkv_gse_brst"]
                    .coords["time"]
                    .values
                )
            else:
                raise NotImplementedError(
                    f"No definition found for PRODUCT == Product.{PRODUCT.upper()}"
                )

    def interp(dat, finite_mask):
        log.warning(
            f"MISSING DATA {np.size(finite_mask) - np.count_nonzero(finite_mask)} non-finite"
        )
        log.info("Correcting missing through interpolation")
        return np.interp(time, time[finite_mask], dat[finite_mask])

    if len(np.shape(data)) > 1:
        log.info("Multi-dimensional data")
        for i in range(np.shape(data)[1]):
            log.info(f"looking at dimension {i}")
            dat = data[:, i]
            finite_mask = np.isfinite(dat)
            if np.size(dat) - np.sum(finite_mask) > 0:
                log.info("nans present")
                data[:, i] = interp(dat, finite_mask)
    else:
        log.info("single-dimension data")
        finite_mask = np.isfinite(data)
        if (np.size(data) - np.sum(finite_mask)) > 0:
            log.info("nans present")
            data = interp(data, finite_mask)

    log.info("Saving arrays")
    np.save(
        f"{dir_name}/data{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy",
        data,
    )
    log.info(
        f"Saved {dir_name}/data{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy"
    )
    np.save(
        f"{dir_name}/time{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy",
        time,
    )
    log.info(
        f"Saved {dir_name}/time{'_' + PRODUCT if PRODUCT != '' else ''}{'_' + SPECIES if SPECIES != '' else ''}.npy"
    )

    del data, time


if __name__ == "__main__":
    # trange = ["2018-03-13/04:41:34", "2018-03-13/04:55:34"]
    # trange = ["2018-03-16/01:39:54", "2018-03-16/01:56:42"]
    # trange = ["2020-03-18/02:57:00", "2020-03-18/03:08:41"]
    with open(get_path(__file__) + "/summary.json", "r") as file:
        summary = json.load(file)
    trange = summary["trange"]
    print(trange)

    # download_data(
    #     trange=trange,
    #     INSTRUMENT=Instrument.FGM,
    #     PRODUCT=Product.R_GSE,
    # )
    # download_data(
    #     trange=trange,
    #     INSTRUMENT=Instrument.FPI,
    #     SPECIES="e",
    #     PRODUCT=Product.BULKV,
    #     data_rate="aftr",
    #     added_time=6,
    # )
    # download_data(
    #     trange=trange,
    #     INSTRUMENT=Instrument.FPI,
    #     SPECIES="i",
    #     PRODUCT=Product.NUMBERDENSITY,
    #     data_rate="aftr",
    #     added_time=6,
    # )
    # download_data(  # FGM SW data
    #     trange=trange,
    #     INSTRUMENT=Instrument.FGM,
    #     SPECIES="",
    #     PRODUCT="",
    #     data_rate="aftr",
    #     added_time=6,
    # )
    # download_data(
    #     trange=trange,
    #     INSTRUMENT=Instrument.FSM,
    #     SPECIES="",
    #     PRODUCT="",
    # )
    # download_data(
    #     trange=trange,
    #     INSTRUMENT=Instrument.FGM,
    #     SPECIES="",
    #     PRODUCT="",
    # )
    # for PRODUCT in [Product.NUMBERDENSITY, Product.TEMPPERP, Product.BULKV]:
    #     for SPECIES in ["i", "e"]:
    #         download_data(
    #             trange=trange,
    #             INSTRUMENT=Instrument.FPI,
    #             SPECIES=SPECIES,
    #             PRODUCT=PRODUCT,
    #         )
