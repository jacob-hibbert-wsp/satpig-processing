# -*- coding: utf-8 -*-
"""Post-processing and formatting on the VKMs outputs."""

##### IMPORTS #####

import itertools
import logging
import pathlib
import warnings
from typing import Literal, Optional

import pandas as pd
import pydantic
from carbon_vkms import utils

import caf.toolkit as ctk

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_CONFIG_FILE = pathlib.Path("vkms_post.yml")

INDEX_COLUMNS = ["origin", "destination", "through"]
OD_DATA_COLUMNS = INDEX_COLUMNS[:2] + [
    "weighted_mean_route_distance",
    "weighted_mean_route_speed",
    "total_abs_demand",
    "total_route_vkms",
]
OD_COLUMNS_RENAME = {
    "weighted_mean_route_distance": "Distance (km)",
    "weighted_mean_route_speed": "Speed (kph)",
    "total_abs_demand": "Trips",
    "total_route_vkms": "VKMs",
}
_DISTANCE_BANDS = [
    "< 1.6km",
    "1.6 - 8.0km",
    "8.0 - 16.1km",
    "16.1 - 40.2km",
    "40.2 - 80.5km",
    ">= 80.5km",
]
_BAND_DATA_NAMES = ["abs_demand", "through_vkms"]
ODT_DATA_COLUMNS = INDEX_COLUMNS + [
    f"total_{i} - {j}" for i, j in itertools.product(_DISTANCE_BANDS, _BAND_DATA_NAMES)
]
ODT_COLUMNS_RENAME = {
    f"total_{i} - {j}": f"{i} - {j}".replace("abs_demand", "Trips").replace(
        "through_vkms", "VKMs"
    )
    for i, j in itertools.product(_DISTANCE_BANDS, _BAND_DATA_NAMES)
}


##### CLASSES & FUNCTIONS #####


class _Config(ctk.BaseConfig):
    vkms_folders: list[pydantic.DirectoryPath]
    output_folder: pydantic.DirectoryPath
    through_zone_lookup: Optional[pydantic.FilePath] = None


def _drop_negative_zones(data: pd.DataFrame, name: str) -> pd.DataFrame:
    original_length = len(data)
    dropped = {}
    for col in data.index.names:
        before = len(data)
        data = data.drop(-1, axis=0, level=col)

        dropped[col] = before - len(data)

    msg = "\n  ".join(
        f"{i:>11.11} : {j:,} ({j / original_length:.1%})" for i, j in dropped.items()
    )
    LOG.warning(
        "Dropped -1 values from columns:\n  %s\n%s rows remaining in %s dataset",
        msg,
        f"{len(data):,}",
        name,
    )

    return data


def _data_summary(data: pd.DataFrame, name: str) -> pd.DataFrame:
    summary: pd.DataFrame = data.describe()
    summary.loc["length", :] = len(data)
    summary.loc["#_nans", :] = data.isna().sum()
    summary = summary.T

    summary.index = pd.MultiIndex.from_tuples(
        [(name, i) for i in summary.index], names=["Name", "Metric"]
    )

    return summary


def process(
    path: pathlib.Path,
    output_folder: pathlib.Path,
    through_zones_lookup: pathlib.Path,
    output_mode: Literal["csv", "excel"] = "csv",
):
    od_path = path.with_name(path.stem + "-OD_VKMs.csv")
    through_path = path.with_name(path.stem + "-ODT_VKMs.csv")

    missing = [f'"{i.name}"' for i in (od_path, through_path) if not i.is_file()]
    if len(missing) > 0:
        raise FileNotFoundError(
            f'cannot find VKMs files ({" and ".join(missing)}) in folder:\n"{path}"'
        )

    LOG.info("Reading %s", od_path.name)
    od_data = pd.read_csv(od_path, index_col=INDEX_COLUMNS[:2], usecols=OD_DATA_COLUMNS)
    od_data = _drop_negative_zones(od_data, "OD")
    od_data = od_data.rename(columns=OD_COLUMNS_RENAME)

    LOG.info("Reading %s", through_path.name)
    through_data = pd.read_csv(through_path, index_col=INDEX_COLUMNS, usecols=ODT_DATA_COLUMNS)
    through_data = _drop_negative_zones(through_data, "OD & Through")
    through_data = through_data.rename(columns=ODT_COLUMNS_RENAME)

    LOG.info("Calculating summary and reformatting")
    summary = pd.concat(
        [_data_summary(od_data, "OD"), _data_summary(through_data, "ODT")], axis=0
    )

    pivoted = through_data.unstack(level="through").reorder_levels([1, 0], axis=1)
    pivoted = pivoted.sort_index(axis=1, sort_remaining=False)

    if through_zones_lookup is not None:
        # TODO Column names should be module constants
        lookup = pd.read_csv(
            through_zones_lookup, usecols=["lad", "through_name"], index_col="lad"
        )
        pivoted = pivoted.rename(columns=lookup["through_name"].to_dict(), level=0)

    od_data.columns = pd.MultiIndex.from_tuples([("OD", i) for i in od_data.columns])
    combined = od_data.merge(
        pivoted, how="outer", validate="1:1", left_index=True, right_index=True
    )

    # Writing to Excel is really slow, so try outputting to CSVs
    timer = utils.Timer()
    LOG.info("Producing VKMs output")
    if output_mode == "csv":
        out_path = output_folder / f"{path.stem}-VKMs_summary.csv"
        summary.to_csv(out_path)
        LOG.info('Written in %s "%s"', timer.time_taken(True), out_path)

        out_path = output_folder / f"{path.stem}-VKMs.csv"
        combined.to_csv(out_path)
        LOG.info('Written in %s "%s"', timer.time_taken(), out_path)

    elif output_mode == "excel":
        warnings.warn("Writing dataset to Excel is very slow with large data")

        out_path = output_folder / f"{path.stem}-VKMs.xlsx"
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as excel:
            summary.to_excel(excel, sheet_name="Summary")
            combined.to_excel(excel, sheet_name="VKMs")

        LOG.info('Written to Excel in %s - "%s"', timer.time_taken(), out_path)

    else:
        raise ValueError(
            f"unexpected value for output_mode ('{output_mode}') should be 'csv' or 'excel'"
        )


def main() -> None:
    parameters = _Config.load_yaml(_CONFIG_FILE)

    details = ctk.ToolDetails("VKMs-post_processing", "0.1.0")
    log_file = parameters.output_folder / (_CONFIG_FILE.stem + ".log")

    with ctk.LogHelper("", details, log_file=log_file):

        timer = utils.Timer()
        for i, folder in enumerate(parameters.vkms_folders, start=1):
            LOG.info(
                "Post Processing VKMs folder %s (%s / %s)",
                folder.name,
                i,
                len(parameters.vkms_folders),
            )
            output_folder = parameters.output_folder / folder.name
            output_folder.mkdir(exist_ok=True)

            file_paths = list(folder.glob("*.h5"))
            for j, path in enumerate(file_paths, start=1):
                process(
                    path,
                    output_folder,
                    through_zones_lookup=parameters.through_zone_lookup,
                    output_mode="csv",
                )
                LOG.info(
                    'Done "%s" %s / %s (%s)',
                    path.name,
                    j,
                    len(file_paths),
                    f"{j/len(file_paths):.0%}",
                )

            LOG.info(
                "Done VKMs folder %s / %s (%s) in %s",
                i,
                len(parameters.vkms_folders),
                f"{i / len(parameters.vkms_folders):.0%}",
                timer.time_taken(True),
            )


##### MAIN #####
if __name__ == "__main__":
    main()
