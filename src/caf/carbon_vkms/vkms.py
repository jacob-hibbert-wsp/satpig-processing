# -*- coding: utf-8 -*-
"""
    Looking at the SATPIG H5 files.
"""

##### IMPORTS #####

import itertools
import logging
import math
import pathlib
import shutil
import sqlite3
import time
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import tqdm

from carbon_vkms import utils

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
SATPIG_HDF_GROUPS = {"od": "data/OD", "routes": "data/Route", "links": "data/link"}
LINK_DATA_COLUMNS = ["speed", "distance"]
LINK_NODE_COLUMNS = ["a", "b"]
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


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def reset(self) -> None:
        self.start = time.perf_counter()

    def time_taken(self, reset: bool = False) -> str:
        time_taken = time.perf_counter() - self.start
        if reset:
            self.reset()

        if time_taken < 60:
            return f"{time_taken:.1f} secs"

        mins, secs = divmod(time_taken, 60)
        if mins < 60:
            return f"{mins:.0f} mins {secs:.0f} secs"

        hours, mins = divmod(mins, 60)
        return f"{hours:.0f}:{mins:.0f}:{secs:.0f}"


def basic_read_df(path):
    print(f"Reading {path.name}")
    timer = Timer()
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

    timer = Timer()
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
    timer = Timer()
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


def update_links_data(store: pd.HDFStore, link_path: pathlib.Path) -> pd.Series:
    LOG.info("Loading SATPig links data from group '%s'", SATPIG_HDF_GROUPS["links"])
    links = store.get(SATPIG_HDF_GROUPS["links"])

    # Join links data and zones to links
    LOG.info("Loading extra links data from '%s'", link_path.name)
    lookup = pd.read_csv(
        link_path,
        index_col=LINK_NODE_COLUMNS,
        usecols=[*LINK_NODE_COLUMNS, "zone", *LINK_DATA_COLUMNS],
    )

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
        LOG.error(
            "Nan values found in links data after joining lookup:\n%s",
            links[lookup.columns].isna().sum(),
        )
        # TODO Raise error if columns have missing values
        LOG.warning("Filling Nan values with -1")
        links["zone"] = links["zone"].fillna(-1)

    # Convert metres to km
    links["distance"] = links["distance"] / 1000

    store.put(SATPIG_HDF_GROUPS["links"], links, format="fixed", complevel=1)
    LOG.info("Updated links data in group '%s'", SATPIG_HDF_GROUPS["links"])

    return links["zone"].astype(int)


def routes_by_zone(store: pd.HDFStore, zone_lookup: pd.Series) -> pd.DataFrame:
    # Produce route zones table with columns: route id, origin zone, destination zone, through zone
    LOG.info("Creating routes zone groupings table")
    timer = Timer()
    routes = store.get(SATPIG_HDF_GROUPS["routes"])
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
        store, key="data/route_zones", complevel=1, format="fixed"
    )

    LOG.info("Generated table in %s, updating file", timer.time_taken(True))

    return route_zones


def _aggregate_final_grouping(
    ungrouped: pd.DataFrame,
    agg_columns: list[str],
    route_columns: list[str],
    banding_columns: list[str],
    output_path: pathlib.Path,
    **csv_kwargs,
) -> pd.DataFrame:
    LOG.info("Performing %s aggregation", ", ".join(agg_columns))

    def demand_weighted_mean(data: pd.Series) -> float:
        return np.average(data, weights=ungrouped.loc[data.index, "abs_demand"])

    agg_methods = {
        "distance": [demand_weighted_mean, "sum"],
        "abs_demand": "sum",
        "through_vkms": "sum",
        "route_vkms": demand_weighted_mean,
        "speed": demand_weighted_mean,
        **dict.fromkeys(route_columns, demand_weighted_mean),
        **dict.fromkeys(banding_columns, "sum"),
    }

    # Aggregate to just origin, destination (weighted avg. based on demand)
    aggregation = ungrouped.groupby(agg_columns).agg(agg_methods)
    aggregation.columns = [
        f"{j}_{i}" if i == "distance" else i for i, j in aggregation.columns
    ]
    aggregation.rename(columns=_VKMS_OUTPUT_RENAMING, inplace=True)

    aggregation.to_csv(output_path, **csv_kwargs)
    LOG.info("Written: %s", output_path)

    return aggregation


def _aggregate_route_zones(
    store: pd.HDFStore,
    route_zones: pd.DataFrame,
    zones: np.ndarray,
    through_lookup: dict[int, int],
    output_path: pathlib.Path,
    header: bool = True,
):
    if len(zones) <= 20:
        zone_str = ", ".join(str(i) for i in zones)
    else:
        zone_str = (
            ", ".join(str(i) for i in zones[:5])
            + " ... "
            + ", ".join(str(i) for i in zones[-5:])
        )

    LOG.info("Processing chunk containing %s zones: %s", len(zones), zone_str)

    if route_zones["done_row"].all():
        LOG.info("No routes left to do")
        return

    # Get mask for all routes which contain the zones
    routes_mask = (
        (route_zones["origin"].isin(zones))
        | (route_zones["destination"].isin(zones))
        | (route_zones["through"].isin(zones))
    )
    # Filter out any o, d, t routes that have already been completed in a previous chunk
    routes_mask.loc[route_zones["done_row"].values] = False

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

    path = output_path.with_name(output_path.stem + "-route_summary.csv")
    route_totals.to_csv(path, **csv_kwargs)
    LOG.info("Written: %s", path)

    def distance_weighted_mean(data: pd.Series) -> float:  # pylint: disable=function-redefined
        return np.average(data, weights=route_data.loc[data.index, "distance"])

    aggregation = route_data.groupby(["route_id", "origin", "destination", "through"])[
        LINK_DATA_COLUMNS
    ].aggregate({"speed": distance_weighted_mean, "distance": "sum"})

    # Join Route totals
    aggregation = aggregation.merge(
        route_totals, how="left", left_index=True, right_index=True, validate="m:1"
    )

    # Join demand data
    od = store.get(SATPIG_HDF_GROUPS["od"])["abs_demand"]
    od.index = od.index.droplevel([i for i in od.index.names if i != "route"])
    od.index.name = "route_id"
    aggregation = aggregation.merge(
        od, how="left", validate="1:1", left_index=True, right_index=True
    )

    aggregation["through_vkms"] = aggregation["abs_demand"] * aggregation["distance"]
    aggregation["route_vkms"] = aggregation["abs_demand"] * aggregation["route_distance"]

    # Create banding columns
    banding_columns = []
    for i, j in zip(_VKM_DISTANCE_BINS[:-1], _VKM_DISTANCE_BINS[1:]):
        if i == -np.inf:
            name = f"< {j}km"
        elif j == np.inf:
            name = f">= {i}km"
        else:
            name = f"{i} - {j}km"

        mask = (aggregation["route_distance"] >= i) & (aggregation["route_distance"] < j)
        col_name = f"{name} - trips"
        aggregation.loc[mask, col_name] = aggregation.loc[mask, "abs_demand"]
        banding_columns.append(col_name)

        col_name = f"{name} - route_vkms"
        aggregation.loc[mask, col_name] = (
            aggregation.loc[mask, "route_distance"] * aggregation.loc[mask, "abs_demand"]
        )
        banding_columns.append(col_name)

        col_name = f"{name} - through_vkms"
        aggregation.loc[mask, col_name] = (
            aggregation.loc[mask, "distance"] * aggregation.loc[mask, "abs_demand"]
        )
        banding_columns.append(col_name)

    if VERBOSE_OUTPUTS:
        path = output_path.with_name(output_path.stem + "-routes.csv")
        aggregation.to_csv(path, **csv_kwargs)
        LOG.info("Written: %s", path)

    _aggregate_final_grouping(
        aggregation,
        ["origin", "destination"],
        route_totals.columns.tolist(),
        banding_columns,
        output_path.with_name(output_path.stem + "-od.csv"),
        **csv_kwargs,
    )

    # Convert through zones, assuming all zones are completely within another i.e many-to-one
    aggregation = aggregation.reset_index()
    if len(through_lookup) == 0:
        LOG.info("Through zones using same zoning as origin and destination")

    else:
        missing = aggregation.loc[
            ~aggregation["through"].isin(through_lookup), "through"
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
            through_lookup,
        )
        aggregation["through"] = aggregation["through"].replace(through_lookup)

    _aggregate_final_grouping(
        aggregation,
        ["origin", "destination", "through"],
        route_totals.columns.tolist(),
        banding_columns,
        output_path,
        **csv_kwargs,
    )

    route_zones.loc[routes_mask.values, "done_row"] = True
    return route_zones


def process_hdf(
    path: pathlib.Path,
    links_data_path: pathlib.Path,
    working_directory: pathlib.Path,
    through_lookup_path: Optional[pathlib.Path] = None,
    chunk_size: int = 100,
    zone_filter: Optional[Sequence[int]] = None,
) -> None:
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

    LOG.info("Reading: %s", path.name)
    with pd.HDFStore(path, "r+") as store:
        LOG.info(str(store.info()))

        zone_lookup = update_links_data(store, links_data_path)
        route_zones = routes_by_zone(store, zone_lookup)
        route_zones["done_row"] = False

        if zone_filter is None:
            zones = np.sort(zone_lookup.unique())
        else:
            zones = np.sort(zone_filter)
        n_chunks = math.ceil(len(zones) / chunk_size)

        # Process chunk of zones
        for i, chunk in enumerate(itertools.batched(zones, chunk_size), start=1):
            route_zones = _aggregate_route_zones(
                store,
                route_zones,
                np.array(chunk),
                through_lookup,
                working_directory / f"{path.stem}_aggregated.csv",
                header=i == 1,
            )
            LOG.info("Done chunk %s / %s (%s)", i, n_chunks, f"{i / n_chunks:.0%}")
