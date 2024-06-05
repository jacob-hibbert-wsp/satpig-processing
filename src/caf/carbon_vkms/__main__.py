# -*- coding: utf-8 -*-
"""Carbon VKMs package."""

##### IMPORTS #####

import datetime as dt
import logging
import pathlib
import warnings

import caf.toolkit as ctk
import pandas as pd

from carbon_vkms import utils, vkms

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def main() -> None:
    warnings.formatwarning = utils.simple_warning_format

    # TODO Move input parameters to config class (YAML file)
    inputs_folder = pathlib.Path(r"B:\QCR- assignments\03.Assignments\h5files\BaseYearFiles")
    output_folder = pathlib.Path(r"B:\QCR- assignments\03.Assignments\h5files\outputs")
    working_directory = output_folder / f"Test VKMs-{dt.datetime.today():%Y%m%d}"
    working_directory.mkdir(exist_ok=True)
    log_file = working_directory / "satpig_tests.log"

    links_data_path = inputs_folder / "2018_link_table_new_2.csv"
    lad_lookup_path = inputs_folder / "MSOA11_WD21_LAD21_EW_LU_1.csv"

    filterpath = inputs_folder.parent / r"YNY\MSOA11_WD21_LAD21_EW_LU_YNY_CA.csv"
    zone_filter = pd.read_csv(filterpath, usecols=["zone"])["zone"].tolist()

    with ctk.LogHelper("", ctk.ToolDetails("satpig_test", "0.1.0"), log_file=log_file):

        for path in inputs_folder.glob("*.h5"):
            vkms.process_hdf(
                path,
                links_data_path,
                working_directory,
                chunk_size=60,
                zone_filter=zone_filter,
            )

    # TODO Loop through all input HDF files
    # TODO Find correct link lookup and link data
    # TODO Find correct MSOA to LAD lookup


##### MAIN #####
if __name__ == "__main__":
    main()
