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
_CONFIG_FILE = pathlib.Path("carbon_vkms.yml")
#VKMS_IGNORE_FOLDER_EXISTS = utils.getenv_bool("VKMS_IGNORE_FOLDER_EXISTS", False)
VKMS_IGNORE_FOLDER_EXISTS = True


##### CLASSES & FUNCTIONS #####


def main() -> None:
    warnings.formatwarning = utils.simple_warning_format

    params = utils.CarbonVKMConfig.load_yaml(_CONFIG_FILE)

    output_folder = params.output_folder / params.output_folder_name_format.format(
        datetime=dt.datetime.now()
    )
    try:
        # Will not use an existing folder to avoid clashes if
        # process is running concurrently on multiple VMs
        output_folder.mkdir(exist_ok=False, parents=True)

    except FileExistsError as exc:
        msg = f'Output run folder already exists: "{output_folder.absolute()}"'

        if VKMS_IGNORE_FOLDER_EXISTS:
            warnings.warn(
                msg + "\nContinuing outputting to folder because "
                f"env variable {VKMS_IGNORE_FOLDER_EXISTS=}",
                RuntimeWarning,
            )

        else:
            raise SystemExit(
                msg + "\nPlease rename or move this folder before re-running."
                "\nSet environment variable VKMS_IGNORE_FOLDER_EXISTS=true"
                " to ignore this error."
            ) from exc

    log_file = output_folder / "satpig_tests.log"

    with ctk.LogHelper("", ctk.ToolDetails("satpig_test", "0.1.0"), log_file=log_file):
        LOG.info("Loading zone filters from: %s", params.zone_filter_path)
        if params.zone_filter_path is not None:
            zone_filter = pd.read_csv(params.zone_filter_path, usecols=["zone"])["zone"].tolist()
            LOG.info(
                "%s zones included in filter: %s",
                len(zone_filter),
                utils.shorten_list(zone_filter, 10),
            )
        else:
            zone_filter = None
        timer = utils.Timer()
        for scenario in params.scenario_paths:
            LOG.info("Producing VKMs for all SATPIG files in: %s", scenario.folder.resolve())
            # Save each scenario to separate working directory to avoid filename clashes
            working_directory = output_folder / scenario.folder.name
            working_directory.mkdir(exist_ok=True)

            timer.reset()
            for path in scenario.folder.glob("*.h5"):
                LOG.info('Processing "%s"', path)
                try:
                    vkms.process_hdf(
                        path,
                        scenario.link_data,
                        scenario.link_cost,
                        working_directory,
                        through_lookup_path=params.through_zones_lookup,
                        chunk_size=params.chunk_size,
                        zone_filter=zone_filter,
                    )
                except Exception:  # Continuing with other files pylint: disable=broad-except
                    LOG.error('Error producing VKMs for "%s"\n', path.resolve(), exc_info=True)
                else:
                    LOG.info(
                        'Finished processing "%s" in %s\n', path.name, timer.time_taken(True)
                    )


##### MAIN #####
if __name__ == "__main__":
    main()
