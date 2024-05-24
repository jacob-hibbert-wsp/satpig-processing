# -*- coding: utf-8 -*-
"""Carbon VKMs package."""

##### IMPORTS #####

import logging
import warnings
import pathlib
import datetime as dt

import caf.toolkit as ctk

from carbon_vkms import utils, vkms

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def main() -> None:
    warnings.formatwarning = utils.simple_warning_format

    # TODO Move input parameters to config class (YAML file)
    inputs_folder = pathlib.Path("Inputs")
    output_folder = pathlib.Path("Outputs")
    working_directory = output_folder / f"VKMs-{dt.datetime.today():%Y%m%d}"
    working_directory.mkdir(exist_ok=True)
    log_file = working_directory / "satpig_tests.log"

    h5_path = inputs_folder / "NoHAM_QCR_DM_Core_2038_TS1_v107_SatPig_uc2_test.h5"
    links_data_path = inputs_folder / "dummy_links_lookup_and_data.csv"
    lad_lookup_path = inputs_folder / "dummy_lad_lookup.csv"

    with ctk.LogHelper("", ctk.ToolDetails("satpig_test", "0.1.0"), log_file=log_file):
        vkms.process_hdf(
            h5_path,
            links_data_path,
            working_directory,
            lad_lookup_path,
            zone_filter=list(range(10)),
        )

    # TODO Loop through all input HDF files
    # TODO Find correct link lookup and link data
    # TODO Find correct MSOA to LAD lookup


##### MAIN #####
if __name__ == "__main__":
    main()
