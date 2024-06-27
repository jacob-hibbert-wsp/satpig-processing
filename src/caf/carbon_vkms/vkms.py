# -*- coding: utf-8 -*-
"""
    Looking at the SATPIG H5 files.
"""

##### IMPORTS #####
from __future__ import annotations
import itertools
import logging
import math
import pathlib
import shutil
import sqlite3
import warnings
from typing import Optional, Sequence

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tqdm
from caf.carbon_vkms import utils

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
SATPIG_HDF_GROUPS = {"od": "data/OD", "routes": "data/Route", "links": "data/link"}
LINK_DATA_COLUMNS = ["speed", "distance"]
LINK_NODE_COLUMNS = ["a", "b"]
_LINKS_FILLNA = {"zone": -1, "speed": 48.0, "distance": 1.0}
_VKMS_OUTPUT_RENAMING = {
    "origin": "origin_zone_id",
    "destination": "destination_zone_id",
    "through": "through_zone_id",
    "abs_demand": "trips",
    "sum_distance": "distance_kms",
    "through_kms": "vehicle_kms",
    "speed": "speed_kph",
}
_VKM_DISTANCE_BINS = (-np.inf, 1.6, 8.0, 16.1, 40.2, 80.5, np.inf)

VERBOSE_OUTPUTS = utils.getenv_bool("CARBON_VKMS_VERBOSE", False)


##### CLASSES & FUNCTIONS #####


def basic_read_df(path):
    print(f"Reading {path.name}")
    timer = utils.Timer()
    df = pd.read_hdf(path)
    print(f"Done reading in {timer.time_taken()}")
    print(df)

    timer.reset()
    print("Grouping links")
    grouped = df.reset_index().groupby(["a", "b"])["abs_demand", "pct_demand"].sum()
    print(f"Done grouping in {timer.time_taken()}")
    print(grouped)


def chunk_fixed(
    path: pathlib.Path,
    out_path: pathlib.Path,
    group_length: int,
    overwrite: bool = True,
):
    if out_path.is_file() and overwrite:
        out_path.unlink()
    elif out_path.is_file():
        raise FileExistsError(f"{out_path.name} already exists and overwrite is False")

    timer = utils.Timer()
    print(f"Reading: {path.name}")
    full = pd.read_hdf(path)
    print(f"Done reading in {timer.time_taken()}")

    origins = full.index.get_level_values("o").unique().sort_values()
    start_index = 0
    timer.reset()
    for end_index in tqdm.trange(
        group_length,
        len(origins) + 1,
        group_length,
        desc="Writing chunks",
        dynamic_ncols=True,
    ):
        origin_indices = origins[start_index:end_index]
        full.loc[pd.IndexSlice[:, origin_indices], :].to_hdf(
            out_path,
            key=f"origin_{start_index}_{end_index}",
            complevel=1,
            format="fixed",
        )
        start_index = end_index

    print(f"Done writing in {timer.time_taken()}")


def convert_to_sqlite(path: pathlib.Path, out_path: pathlib.Path):
    timer = utils.Timer()
    print(f"Reading: {path.name}")
    full = pd.read_hdf(path)
    full.index = full.index.droplevel("n_link")
    # Avoid writing the index to see if this speeds up write
    full.reset_index(inplace=True)
    print(f"Done reading in {timer.time_taken()}")

    if out_path.suffix not in (".sqlite", ".db"):
        raise ValueError(f"invalid suffix for output path: {out_path.suffix!r}")

    timer.reset()
    with sqlite3.connect(out_path) as conn:
        start_index = 0
        nrows = 10_000_000

        for end_index in tqdm.trange(
            nrows, len(full) + 1, nrows, desc="Writing to SQLite", dynamic_ncols=True
        ):
            full.iloc[start_index:end_index, :].to_sql(
                path.stem,
                conn,
                if_exists="fail" if start_index == 0 else "append",
                index=False,
            )
            start_index = end_index

    print(f"Written to {out_path.name} in {timer.time_taken()}")


def _check_merge_indicator(
    data: pd.DataFrame, left_name: str, right_name: str, drop: bool = True
) -> None:
    if (data["_merge"] == "both").all():
        LOG.info("Merged %s into %s and found all rows in both", left_name, right_name)
        if drop:
            data.drop(columns="_merge", inplace=True)

        return

    merge_column = data["_merge"].cat.rename_categories(
        {"left_only": f"{left_name} only", "right_only": f"{right_name} only"}
    )
    counts = dict(np.column_stack(np.unique(merge_column, return_counts=True)))

    msg = f"Merged {right_name} into {left_name} and found rows in: " + ", ".join(
        f"{i} = {int(j):,} ({int(j) / len(merge_column):.1%})" for i, j in counts.items()
    )

    if set(counts.keys()) != {"both"}:
        warnings.warn(msg, RuntimeWarning)
    else:
        LOG.info(msg)

    if drop:
        data.drop(columns="_merge", inplace=True)


def update_links_data(store: pd.HDFStore, link_path: pathlib.Path, cost_path: pathlib.Path) -> pd.Series:
    LOG.info("Loading SATPig links data from group '%s'", SATPIG_HDF_GROUPS["links"])
    links = store.get(SATPIG_HDF_GROUPS["links"])
    for col in LINK_NODE_COLUMNS:
        links[col] = pd.to_numeric(links[col], downcast="integer")

    # Join links data and zones to links
    LOG.info("Loading extra links data from '%s'", link_path.name)
    lookup = pd.read_csv(
        link_path,
        index_col=LINK_NODE_COLUMNS,
        usecols=[*LINK_NODE_COLUMNS, "zone", *LINK_DATA_COLUMNS],
    )

    link_cost = pd.read_csv(cost_path)

    duplicates = lookup.index.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} duplicate links found in links lookup")

    # Load link lookup and set link zones
    links = links.merge(
        lookup,
        left_on=LINK_NODE_COLUMNS,
        right_index=True,
        how="left",
        validate="1:1",
        indicator=True,
    )
    _check_merge_indicator(links, "SATPig", "Links Data")

    ab_duplicates = links.duplicated(LINK_NODE_COLUMNS, keep="first").sum()
    id_duplicates = links.index.duplicated(keep="first").sum()
    if ab_duplicates > 0 or id_duplicates > 0:
        warnings.warn(
            f"found {ab_duplicates:,} duplicate a, b nodes"
            f" and {id_duplicates:,} duplicate link IDs",
            RuntimeWarning,
        )

    if links[lookup.columns].isna().any(axis=None):
        warnings.warn(
            "Nan values found in links data after joining"
            f" lookup:\n{links[lookup.columns].isna().sum()}."
            f"\nFilling NaN values with: {_LINKS_FILLNA}",
            RuntimeWarning,
        )
        for nm, value in _LINKS_FILLNA.items():
            links[nm] = links[nm].fillna(value)

        if links[lookup.columns].isna().any(axis=None):
            raise ValueError(
                "Links data still contains Nan values after"
                " infilling, this shouldn't be possible"
            )

    # Convert metres to km
    links["distance"] = links["distance"] / 1000

    store.put(SATPIG_HDF_GROUPS["links"], links, format="fixed", complevel=1)
    LOG.info("Updated links data in group '%s'", SATPIG_HDF_GROUPS["links"])

    return links["zone"].astype(int)


def routes_by_zone(path: pathlib.Path, zone_lookup: pd.Series) -> pd.DataFrame:
    # Produce route zones table with columns: route id, origin zone, destination zone, through zone
    LOG.info("Creating routes zone groupings table")
    timer = utils.Timer()
    routes = pd.read_hdf(path, SATPIG_HDF_GROUPS["routes"])
    LOG.info(
        "Loaded routes in %s, now joining zones this may take some time",
        timer.time_taken(True),
    )

    if zone_lookup.index.has_duplicates:
        dups = zone_lookup.index[zone_lookup.index.duplicated(keep="first")].unique()
        warnings.warn(
            f"{len(dups):,} duplicate zones route IDs found in zone"
            f" lookup: {utils.shorten_list(dups, 10)}",
            RuntimeWarning,
        )

    # Join links to routes to get all routes relevant for single MSOA
    route_zones = routes.merge(
        zone_lookup,
        how="left",
        left_index=True,
        right_index=True,
        validate="m:1",
        indicator=True,
        copy=False,
    ).reset_index()
    LOG.info("Done merge in %s", timer.time_taken(True))
    del routes, zone_lookup
    _check_merge_indicator(route_zones, "routes", "link zones")
    LOG.info("Done merge check in %s", timer.time_taken(True))

    route_zones.drop(columns="link_id", inplace=True, errors="ignore")
    route_zones.drop_duplicates(subset=["route", "zone"], inplace=True)
    LOG.info("Done drop duplicates in %s", timer.time_taken(True))

    route_zones = (
        route_zones.sort_values(["route", "link_order_id"])[["route", "zone"]]
        .groupby("route", as_index=False)
        .agg(["first", "last", tuple])
    )
    route_zones.columns = route_zones.columns.droplevel(0)
    route_zones.rename(
        columns={
            "": "route_id",
            "first": "origin",
            "last": "destination",
            "tuple": "through",
        },
        inplace=True,
    )
    route_zones = route_zones.explode("through")

    errors = []
    for c in route_zones.columns:
        try:
            route_zones[c] = route_zones[c].astype(int)
        except (ValueError, pd.errors.IntCastingNaNError) as exc:
            errors.append(f"column {c!r}: {exc}")

    if len(errors) > 0:
        raise ValueError(
            "cannot convert route zones columns to integers:\n" + "\n".join(errors)
        )

    route_zones.reset_index().to_hdf(
        path, key="data/route_zones", complevel=1, format="fixed", index=False
    )

    LOG.info("Generated table in %s, updating file", timer.time_taken(True))

    return route_zones


def _routes_by_zone_dask(path: pathlib.Path, zone_lookup: pd.Series) -> pd.DataFrame:
    # Implementation of routes_by_zone using dask to avoid memory errors
    # Produce route zones table with columns: route id, origin zone, destination zone, through zone
    LOG.info("Creating routes zone groupings table with dask")
    if zone_lookup.index.has_duplicates:
        dups = zone_lookup.index[zone_lookup.index.duplicated(keep="first")].unique()
        warnings.warn(
            f"{len(dups):,} duplicate zones route IDs found in zone"
            f" lookup: {utils.shorten_list(dups, 10)}",
            RuntimeWarning,
        )

    routes: dd.DataFrame = dd.read_hdf(path, SATPIG_HDF_GROUPS["routes"])

    # Join links to routes to get all routes relevant for single MSOA
    routes = routes.merge(
        zone_lookup,
        how="left",
        left_index=True,
        right_index=True,
        indicator=True,
        # validate="m:1",
    )

    # TODO Log number of values found each side of merge
    # _check_merge_indicator(route_zones, "routes", "link zones")

    routes = routes.drop(columns="link_id", errors="ignore")
    routes = routes.drop_duplicates(subset=["route", "zone"])
    routes = routes.sort_values(["route", "link_order_id"])[["route", "zone"]]
    LOG.info("Performing groupby")

    # TODO I think groupby performs dask compute and returns a dataframe???
    routes: pd.DataFrame = routes.groupby("route").agg(["first", "last", tuple])

    routes.columns = routes.columns.droplevel(0)
    routes.rename(
        columns={
            "": "route_id",
            "first": "origin",
            "last": "destination",
            "tuple": "through",
        },
        inplace=True,
    )
    routes = routes.explode("through")

    errors = []
    for c in routes.columns:
        try:
            routes[c] = routes[c].astype(int)
        except (ValueError, pd.errors.IntCastingNaNError) as exc:
            errors.append(f"column {c!r}: {exc}")

    if len(errors) > 0:
        raise ValueError(
            "cannot convert route zones columns to integers:\n" + "\n".join(errors)
        )

    routes.reset_index().to_hdf(
        path, key="data/route_zones", complevel=1, format="fixed", index=False
    )

    LOG.info("Generated table, updating file")

    return routes


def _aggregate_routes(
    ungrouped_path: pathlib.Path,
    agg_columns: Sequence[str],
    distance_band_column: str,
    weighting_column: str,
    band_columns: Sequence[str],
    output_path: pathlib.Path,
    **csv_kwargs,
) -> pd.DataFrame:
    timer = utils.Timer()
    LOG.info(
        "Reading %s and performing %s aggregation", ungrouped_path.name, ", ".join(agg_columns)
    )
    ungrouped = pd.read_csv(ungrouped_path)
    if ungrouped["route_id"].duplicated().any():
        dups = ungrouped["route_id"].duplicated().sum()
        warnings.warn(f"Found {dups:,} duplicate route IDs in route summary data")

    ungrouped, banding_columns = _add_distance_banding(
        ungrouped, distance_band_column, band_columns
    )

    def weighted_mean(data: pd.Series) -> float:
        return np.average(data, weights=ungrouped.loc[data.index, weighting_column])

    agg_methods = {
        distance_band_column: [weighted_mean, "sum"],
        "route_speed": weighted_mean,
        "abs_demand": "sum",
        "route_vkms": "sum",
    }
    agg_methods.update(dict.fromkeys(banding_columns, "sum"))

    aggregation: pd.DataFrame = ungrouped.groupby(agg_columns).agg(agg_methods)

    if aggregation.columns.nlevels > 1:
        aggregation.columns = [
            f"{j}_{i}".replace("sum_", "total_") for i, j in aggregation.columns
        ]

    aggregation.rename(columns=_VKMS_OUTPUT_RENAMING, inplace=True)

    aggregation.to_csv(output_path, **csv_kwargs)
    LOG.info("Done aggregation in %s and written: %s", timer.time_taken(), output_path)


def _aggregate_through(
    ungrouped_path: pathlib.Path,
    agg_columns: Sequence[str],
    distance_band_column: str,
    weighting_column: str,
    band_columns: Sequence[str],
    output_path: pathlib.Path,
    distance_band_path: pathlib.Path,
    **csv_kwargs,
) -> pd.DataFrame:
    timer = utils.Timer()
    LOG.info(
        "Reading %s and performing %s aggregation", ungrouped_path.name, ", ".join(agg_columns)
    )
    ungrouped = pd.read_csv(ungrouped_path)

    db_data = pd.read_csv(
        distance_band_path, usecols=["route_id", distance_band_column], index_col="route_id"
    )

    ungrouped = ungrouped.merge(
        db_data,
        how="left",
        left_on="route_id",
        right_index=True,
        validate="m:1",
    )

    ungrouped, banding_columns = _add_distance_banding(
        ungrouped, distance_band_column, band_columns
    )

    def weighted_mean(data: pd.Series) -> float:
        return np.average(data, weights=ungrouped.loc[data.index, weighting_column])

    agg_methods = {
        distance_band_column: weighted_mean,
        "distance": [weighted_mean, "sum"],
        "speed": weighted_mean,
        "abs_demand": "sum",
        "through_vkms": "sum",
    }
    agg_methods.update(dict.fromkeys(banding_columns, "sum"))

    aggregation: pd.DataFrame = ungrouped.groupby(agg_columns).agg(agg_methods)

    if aggregation.columns.nlevels > 1:
        aggregation.columns = [
            f"{j}_{i}".replace("sum_", "total_") for i, j in aggregation.columns
        ]

    aggregation.rename(columns=_VKMS_OUTPUT_RENAMING, inplace=True)

    aggregation.to_csv(output_path, **csv_kwargs)
    LOG.info("Done aggregation in %s and written: %s", timer.time_taken(), output_path)


def _add_distance_banding(
    data: pd.DataFrame, distance_column: str, band_columns: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    banding_columns = []
    for i, j in zip(_VKM_DISTANCE_BINS[:-1], _VKM_DISTANCE_BINS[1:]):
        if i == -np.inf:
            name = f"< {j}km"
        elif j == np.inf:
            name = f">= {i}km"
        else:
            name = f"{i} - {j}km"

        mask = (data[distance_column] >= i) & (data[distance_column] < j)

        for col in band_columns:
            col_name = f"{name} - {col}"
            data.loc[mask, col_name] = data.loc[mask, col]
            banding_columns.append(col_name)

    data[banding_columns] = data[banding_columns].fillna(0)
    return data, banding_columns


def _aggregate_route_zones(
    store: pd.HDFStore,
    route_zones: pd.DataFrame,
    zones: np.ndarray,
    through_lookup: dict[int, int],
    output_path: pathlib.Path,
    header: bool = True,
) -> tuple[pd.DataFrame | None, pathlib.Path, pathlib.Path]:
    if len(zones) <= 20:
        zone_str = ", ".join(str(i) for i in zones)
    else:
        zone_str = (
            ", ".join(str(i) for i in zones[:5])
            + " ... "
            + ", ".join(str(i) for i in zones[-5:])
        )

    # TODO Use VKMSOutputPaths as an input parameter to provide output paths
    summary_path = output_path.with_name(output_path.stem + "-route_summary.csv")
    through_path = output_path.with_name(output_path.stem + "-routes_through.csv")
    LOG.info("Processing chunk containing %s zones %s", len(zones), zone_str)

    if route_zones["done_row"].all():
        LOG.info("No routes left to do")
        return None, summary_path, through_path

    # Get mask for all routes which contain the zones
    routes_mask = (
        (route_zones["origin"].isin(zones))
        | (route_zones["destination"].isin(zones))
        | (route_zones["through"].isin(zones))
    )
    # Make sure to include all rows referring to the same route ID
    unique_route_ids = route_zones.loc[routes_mask, "route_id"].unique()
    routes_mask.loc[route_zones["route_id"].isin(unique_route_ids)] = True
    # Filter out any o, d, t routes that have already been completed in a previous chunk
    routes_mask.loc[route_zones["done_row"].values] = False

    LOG.info(
        "Processing %s / %s (%s) rows of routes dataset in this chunk",
        f"{routes_mask.sum():,}",
        f"{len(routes_mask):,}",
        f"{routes_mask.sum() / len(routes_mask):.0%}",
    )

    # Check any route IDs included in the mask aren't in the remaining route zones data
    unique_route_ids = route_zones.loc[routes_mask, "route_id"].unique()
    unique_route_ids = unique_route_ids[
        np.isin(unique_route_ids, route_zones.loc[~routes_mask, "route_id"])
    ]
    if len(unique_route_ids) > 0:
        raise ValueError(
            f"{len(unique_route_ids)} unique route IDs from"
            " the routes mask also found outside the"
            f" mask: {utils.shorten_list(unique_route_ids, 10)}"
        )

    # Get route link data for all routes found in zone chunk
    links = (
        store.get(SATPIG_HDF_GROUPS["routes"])
        .loc[route_zones.loc[routes_mask, "route_id"].unique()]
        .reset_index()
    )
    links = links.merge(
        store.get(SATPIG_HDF_GROUPS["links"])[["zone"] + LINK_DATA_COLUMNS],
        left_on="link_id",
        right_index=True,
        how="left",
        validate="m:1",
    ).set_index(["route", "zone"])

    # Join links data to zones for aggregation
    route_data = (
        route_zones.loc[routes_mask, :]
        .merge(
            links,
            how="left",
            left_on=["route_id", "through"],
            right_index=True,
            validate="1:m",
        )
        .reset_index(drop=True)
    )
    # TODO Check for duplicate routes with same links

    csv_kwargs = dict(header=header, mode="w" if header else "a")
    LOG.info("Writing CSV with kwargs: %s", csv_kwargs)

    if VERBOSE_OUTPUTS:
        path = output_path.with_name(output_path.stem + "-link_data.csv")
        route_data.to_csv(path, index=False, **csv_kwargs)
        LOG.info("Written: %s", path.name)

    def distance_weighted_mean(data: pd.Series) -> float:
        return np.average(data, weights=route_data.loc[data.index, "distance"])

    # Calculate aggregations across whole routes and for specific through zones separately
    route_totals = route_data.groupby(["route_id", "origin", "destination"])[
        LINK_DATA_COLUMNS
    ].aggregate({"speed": distance_weighted_mean, "distance": "sum"})
    route_totals.columns = [f"route_{i}" for i in route_totals.columns]

    if route_totals.index.get_level_values("route_id").has_duplicates:
        dups = route_totals.index.get_level_values("route_id").duplicated().sum()
        warnings.warn(f"Found {dups:,} duplicate route IDs in route summary OD grouping")

    # Join demand data and calculate route VKMs
    od = store.get(SATPIG_HDF_GROUPS["od"])["abs_demand"]
    od.index = od.index.droplevel([i for i in od.index.names if i != "route"])
    od.index.name = "route_id"

    route_totals = route_totals.merge(
        od, how="left", validate="1:1", left_index=True, right_index=True
    )
    route_totals["route_vkms"] = route_totals["route_distance"] * route_totals["abs_demand"]

    route_totals.to_csv(summary_path, **csv_kwargs)
    LOG.info("Written: %s", summary_path)
    del route_totals

    # Convert through zones, assuming all zones are completely within another i.e many-to-one
    route_data = route_data.reset_index()
    if len(through_lookup) == 0:
        LOG.info("Through zones using same zoning as origin and destination")

    else:
        missing = route_data.loc[
            ~route_data["through"].isin(through_lookup), "through"
        ].unique()
        if len(missing) > 0:
            warnings.warn(
                f"{len(missing):,} unique zones in through zones not found"
                f" in lookup: {utils.shorten_list(missing, 20)}",
                RuntimeWarning,
            )

        LOG.info(
            "Converting through zones using lookup with %s values: %s",
            len(through_lookup),
            utils.shorten_list([f"{i}: {j}" for i, j in through_lookup.items()], 10),
        )
        route_data["through"] = route_data["through"].replace(through_lookup)

    routes_through = route_data.groupby(["route_id", "origin", "destination", "through"])[
        LINK_DATA_COLUMNS
    ].aggregate({"speed": distance_weighted_mean, "distance": "sum"})

    routes_through = routes_through.merge(
        od, how="left", validate="1:1", left_index=True, right_index=True
    )
    routes_through["through_vkms"] = routes_through["distance"] * routes_through["abs_demand"]

    routes_through.to_csv(through_path, **csv_kwargs)
    LOG.info("Written: %s", through_path)

    route_zones.loc[routes_mask.values, "done_row"] = True
    return route_zones, summary_path, through_path


class VKMSOutputPaths:
    """Manages creating VKMs output paths."""

    def __init__(self, satpig_path: pathlib.Path, output_folder: pathlib.Path) -> None:
        self.satpig_path = pathlib.Path(satpig_path)
        self.output_folder = pathlib.Path(output_folder)

        if not self.satpig_path.is_file():
            raise FileNotFoundError(f"cannot find SATPIG file: {self.satpig_path}")

        if not self.output_folder.is_dir():
            raise NotADirectoryError(f"output folder doesn't exist: {self.output_folder}")

    @property
    def aggregate_base_path(self) -> pathlib.Path:
        """Base path for full dataset produced during chunked process."""
        return self.output_folder / f"{self.satpig_path.stem}_aggregated.csv"

    @property
    def od_path(self) -> pathlib.Path:
        """Path for final output of OD VKMs."""
        return self.output_folder / f"{self.satpig_path.stem}-OD_VKMs.csv"

    @property
    def through_path(self) -> pathlib.Path:
        """Path for final output of OD-Through VKMs."""
        return self.output_folder / f"{self.satpig_path.stem}-ODT_VKMs.csv"

    def check_vkms_exist(self, inc_aggregate: bool = False) -> bool:
        exist = self.od_path.is_file() and self.through_path.is_file()

        if inc_aggregate and exist:
            exist = exist and self.aggregate_base_path.is_file()

        return exist


def _pre_process_hdf(
    path: pathlib.Path,
    links_data_path: pathlib.Path,
    link_cost_path: pathlib.Path,
    zone_filter: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    LOG.info("Reading: %s", path.name)
    with pd.HDFStore(path, "r+") as store:
        LOG.info(str(store.info()))
        zone_lookup = update_links_data(store, links_data_path, link_cost_path)

    if zone_filter is None:
        zones = np.sort(zone_lookup.unique())
    else:
        zones = np.sort(zone_filter)

    try:
        route_zones = routes_by_zone(path, zone_lookup)
    except MemoryError as exc:
        LOG.error(
            "Ran out of memory performing route zone lookup using"
            " pandas, so attempting to use dask.\n%s: %s",
            exc.__class__.__name__,
            exc,
        )
        route_zones = _routes_by_zone_dask(path, zone_lookup)

    route_zones["done_row"] = False
    return zones, route_zones


def process_hdf(
    path: pathlib.Path,
    links_data_path: pathlib.Path,
    cost_data_path: pathlib.Path,
    working_directory: pathlib.Path,
    through_lookup_path: Optional[pathlib.Path] = None,
    chunk_size: int = 100,
    zone_filter: Optional[Sequence[int]] = None,
) -> None:
    output_paths = VKMSOutputPaths(path, working_directory)
    if output_paths.check_vkms_exist():
        # TODO Add overwrite parameter to reproduce the VKM output
        # regardless of if they already exist
        LOG.info(
            "VKM outputs for %s already exist:\nOD output: %s\nODT output: %s",
            output_paths.satpig_path.name,
            output_paths.od_path,
            output_paths.through_path,
        )
        return

    # Copy HDF file to working directory to make changes
    LOG.info(
        "Copying SATPig HDF file (%s) into working directory: %s", path.name, working_directory
    )
    path = pathlib.Path(shutil.copy2(path, working_directory))

    if through_lookup_path is not None:
        through_lookup_data = pd.read_csv(
            through_lookup_path, usecols=["zone", "lad"], dtype=int, index_col="zone"
        )
        if through_lookup_data.index.has_duplicates:
            raise ValueError("Through lookup has duplicate values in the zone column")

        through_lookup: dict[int, int] = through_lookup_data["lad"].to_dict()

    else:
        through_lookup = {}

    zones, route_zones = _pre_process_hdf(path, links_data_path, cost_data_path, zone_filter)
    n_chunks = math.ceil(len(zones) / chunk_size)

    with pd.HDFStore(path, "r+") as store:
        # Process chunk of zones
        timer = utils.Timer()
        route_summary_path, route_through_path = None, None
        for i, chunk in enumerate(itertools.batched(zones, chunk_size), start=1):
            route_zones, route_summary_path, route_through_path = _aggregate_route_zones(
                store,
                route_zones,
                np.array(chunk),
                through_lookup,
                output_paths.aggregate_base_path,
                header=i == 1,
            )
            LOG.info(
                "Done chunk %s / %s (%s) in %s",
                i,
                n_chunks,
                f"{i / n_chunks:.0%}",
                timer.time_taken(True),
            )

            if route_zones is None:
                break

    if route_summary_path is not None:
        _aggregate_routes(
            route_summary_path,
            ["origin", "destination"],
            "route_distance",
            "abs_demand",
            ["abs_demand", "route_vkms"],
            output_path=output_paths.od_path,
        )
    else:
        warnings.warn("No route summary produce")

    if route_through_path is not None:
        if route_summary_path is None:
            raise ValueError(
                "somehow route through data is produced but"
                " not summary data, this shouldn't be possible"
            )

        _aggregate_through(
            route_through_path,
            ["origin", "destination", "through"],
            "route_distance",
            "abs_demand",
            ["abs_demand", "through_vkms"],
            output_path=output_paths.through_path,
            distance_band_path=route_summary_path,
        )
    else:
        warnings.warn("No route through data produced")
