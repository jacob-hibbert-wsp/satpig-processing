from os import path
import pathlib

import pandas as pd


LINK_DATA_PATH = pathlib.Path(
    r"A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\2018_link_table_fixed.csv"
)

LINK_TO_ZONE_LOOKUP_PATH = pathlib.Path(
    r"G:\TINA\route_skim_inputs\noham_network_noham_spatial.csv"
)

OUTPATH = pathlib.Path(
    r"G:\TINA\route_skim_inputs\link_data.csv"
)


def main(link_data_path: pathlib.Path, link_to_zone_lookup_path: pathlib.Path, outpath: pathlib.Path):
    
    link_data: pd.DataFrame = pd.read_csv(link_data_path, usecols=["a", "b", "speed", "distance"])
    link_to_zone_lookup = pd.read_csv(
        link_to_zone_lookup_path, usecols=["A", "B", "noham_id", "factor"]
    ).rename(columns = {"A": "a", "B": "b", "noham_id":"zone"})

    link_to_zone_lookup = first_past_post_selection(link_to_zone_lookup)

    link_data_zones = link_data.merge(link_to_zone_lookup, on = ["a", "b"], how = "left")

    link_data_zones.to_csv(outpath)

    return


def first_past_post_selection(link_to_zone_lookup: pd.DataFrame) -> pd.DataFrame:
    link_to_zone_lookup_simplified = link_to_zone_lookup.groupby(["a", "b"]).apply(
        lambda subset: subset.loc[subset["factor"].idxmax()]
    )
    return link_to_zone_lookup_simplified.drop(columns=["factor", "a", "b"]).reset_index()


if __name__ == "__main__":
    main(LINK_DATA_PATH, LINK_TO_ZONE_LOOKUP_PATH, OUTPATH)
