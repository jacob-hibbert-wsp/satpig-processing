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
_CONFIG_FILE = pathlib.Path(__package__).with_suffix(".yml")


##### CLASSES & FUNCTIONS #####


def main() -> None:
    warnings.formatwarning = utils.simple_warning_format

    params = utils.CarbonVKMConfig.load_yaml(_CONFIG_FILE)

    output_folder = params.output_folder / f"VKMs-{dt.datetime.now():%Y%m%d}"
    try:
        # Will not use an existing folder to avoid clashes if
        # process is running concurrently on multiple VMs
        output_folder.mkdir(exist_ok=False, parents=True)

    except FileExistsError as exc:
        raise SystemExit(
            f'Output run folder already exists: "{output_folder.absolute()}"'
            "\nPlease rename or move this folder before re-running."
        ) from exc

    log_file = output_folder / "satpig_tests.log"

    with ctk.LogHelper("", ctk.ToolDetails("satpig_test", "0.1.0"), log_file=log_file):
        LOG.info("Loading zone filters from: %s", params.zone_filter_path)
        zone_filter = pd.read_csv(params.zone_filter_path, usecols=["zone"])["zone"].tolist()
        LOG.info(
            "%s zones included in filter: %s",
            len(zone_filter),
            utils.shorten_list(zone_filter, 10),
        )

        for scenario in params.scenario_paths:
            LOG.info("Producing VKMs for all SATPIG files in: %s", scenario.folder.resolve())
            # Save each scenario to separate working directory to avoid filename clashes
            working_directory = output_folder / scenario.folder.name
            working_directory.mkdir(exist_ok=True)

            for path in scenario.folder.glob("*.h5"):
                try:
                    vkms.process_hdf(
                        path,
                        scenario.link_data,
                        working_directory,
                        through_lookup_path=params.through_zones_lookup,
                        chunk_size=params.chunk_size,
                        zone_filter=zone_filter,
                    )
                except Exception:  # Continuing with other files pylint: disable=broad-except
                    LOG.error('Error producing VKMs for "%s"', path.resolve(), exc_info=True)


##### MAIN #####
if __name__ == "__main__":
    main()
