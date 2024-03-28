# -*- coding: utf-8 -*-
"""
Created on: 2/5/2024
Updated on:

Original author: Matteo Gravellu
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position


# TLD CONSTANTS
MS_TO_MILES = 0.000621371

TLD_RANGES = pd.DataFrame({
    'distance_range': ['1','2', '3', '5', '10', '15', '25', '35', '50', '100', '200'],
    'min_distance': [0, 1, 2, 3, 5, 10, 15, 25, 35, 50, 100],
    'max_distance': [1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200]
})

# temporary
# satpig_path = r"G:\raw_data\4019 - road OD flows\Satpig\QCR\2018\RotherhamBase_i8c_2018_TS2_v107_SatPig_uc1.h5"
# cafspace_path = r"G:\raw_data\caf.space\noham_link_to_lta\noham2018_lta.csv"
# noham_lookup = r"G:\raw_data\caf.space\noham_to_lta\noham_lta_spatial.csv"
# p1xdump_path = r"G:\raw_data\archive\p1xdump\QCR\2018\link_table_NoHAM_Base_2018_TS1_v106_I6.csv"
# study_area_id = rf"West Yorkshire CA"
# output_path = r"G:\raw_data\6001 - under processing\templates"


def assign_cut_range(df, distance_col, new_column):
    """
    Assigns a distance range to each distance value in the DataFrame based on predefined ranges.

    Parameters:
        df (DataFrame): The input DataFrame.
        distance_col (str): The name of the distance column in the DataFrame.
        new_column (str): The name of the new column in the Dataframe with the distance band.

    Returns:
        DataFrame: The DataFrame with an additional column 'Distance Range' containing the assigned distance ranges.
    """
    # Define distance ranges
    ranges = [0, 1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200]
    labels = ['1', '2', '3', '5', '10', '15', '25', '35', '50', '100', '200']

    # Assign distance range based on distance column
    df[new_column] = pd.cut(df[distance_col], bins=ranges,labels=labels, right=False)

    return df


def calculate_percentage(df, value_col, group_cols, perc_column):
    """
    Calculates the percentage of each value within groups defined by other columns.

    Parameters:
        df (DataFrame): The input DataFrame.
        value_col (str): The name of the column containing values for which to calculate percentages.
        group_cols (list): A list of column names to group by.
        perc_column (str): The name of the new column containing percentages

    Returns:
        DataFrame: The DataFrame with an additional column 'Percentage' containing the calculated percentages.
    """

    if not group_cols:
        # Calculate sums within each group
        sum_value = df[value_col].sum()

        # Calculate percentages
        df[perc_column] = (df[value_col] / sum_value)
    else:
        # Calculate sums within each group
        grouped_sums = df.groupby(group_cols)[value_col].transform('sum')

        # Calculate percentages
        df[perc_column] = (df[value_col] / grouped_sums)

    return df

def merge_dataframes(df1: pd.DataFrame, 
                     df2: pd.DataFrame, 
                     on_list: list,
                     how_method = str):
    """
    Merge two DataFrames based on specified parameters.

    Parameters:
    - df1 (DataFrame): First DataFrame to merge.
    - df2 (DataFrame): Second DataFrame to merge.
    - on_list (list): Column(s) to join on. If not specified, uses all common columns.
    - how (str, optional): Type of merge to perform ('inner', 'left', 'right', 'outer').

    Returns:
    - merged_df (DataFrame): Merged DataFrame.
    """
    merged_df = pd.merge(df1, df2, on=on_list, how=how_method)
    return merged_df


def routing_pattern(satpig_path: str,
                    cafspace_path: str,
                    noham_lookup: str,
                    p1xdump_path: str,
                    study_area_id: str,
                    ):
    # read satpig routing input
    # extract the o-d-route based demand and keep it as a separate dataframe
    # get rid of the demand and proceed with routing analysis
    #TODO: is reading the whole file optimal? Can this process be chunked?
    # The satpig at the moment is saved in not table format hdf5,
    # so we might have to go back to the conversion process if a reading filter might help.

    print(rf"Working on: {satpig_path}")
    print(rf"with {cafspace_path} --- rf{noham_lookup} --- {p1xdump_path}")
    file = pd.read_hdf(satpig_path, key='test', mode='r')
    file = file.drop(columns=['pct_demand'])
    file = file.droplevel(['n_link','order'])

    file_demand = file.droplevel(['a', 'b'])
    file_demand = file_demand[~file_demand.index.duplicated(keep='first')]

    file = file.reset_index(level=['o', 'd'])
    file = file.drop(columns=['abs_demand'])


    # read link to output area lookup
    lookup = pd.read_csv(cafspace_path)
    lookup = lookup.set_axis(['a', 'b', 'area', 'factor'], axis=1, copy=False)
    if isinstance(study_area_id, str):
        lookup_area = lookup[lookup['area'] == study_area_id]   
        lookup_check = lookup_area[['a', 'b']]
        lookup_check['check'] = 1
    if isinstance(study_area_id, list):
        lookup_area = lookup.loc[lookup['area'].isin(study_area_id)]
        lookup_check = lookup_area[['a', 'b']]
        lookup_check['check'] = 1
    # read modelling zone to output area lookup
    noham = pd.read_csv(noham_lookup)
    noham = noham.iloc[:, [i for i in range(len(noham.columns)) if i not in [0, 3, 4]]]
    noham = noham.set_axis(['area','zone'], axis=1, copy=False)
    if isinstance(study_area_id, str):
        noham_area = noham[noham['area'] == study_area_id]
        noham_list = noham_area['zone'].tolist()
    if isinstance(study_area_id, list):
        noham_area = noham.loc[noham['area'].isin(study_area_id)]
        noham_list = noham_area['zone'].tolist()  
    # read p1xdump file
    link_distance = pd.read_csv(p1xdump_path, usecols=['A', 'B', 'Distance'])
    link_distance = link_distance.set_axis(['a', 'b', 'distance'], axis=1, copy=False)
    link_distance['distance'] = link_distance['distance'].astype(int)

    print(rf"Input uploaded and route checks started.")

    # Check the routing pattern of the routes (internal, from, to, through)
    #TODO: this next section is quite fast, is there any other filtering code alternative that might even be faster?

    file_ii = file[np.logical_and(file['o'].isin(noham_list),
                                  file['d'].isin(noham_list))]

    file_ie = file[np.logical_and(file['o'].isin(noham_list),
                                  ~file['d'].isin(noham_list))]

    file_ei = file[np.logical_and(~file['o'].isin(noham_list),
                                  file['d'].isin(noham_list))]

    file_ee = file[np.logical_and(~file['o'].isin(noham_list),
                                  ~file['d'].isin(noham_list))]

    del file
    
    file_ee = file_ee.reset_index(level=['a','b','route'])

    # Understand if a route in the ee dataframe is through the dedicated area
    #TODO: this obj to int conversion is slow, but required as a and b in the lookup are integers.
    # Should we do the opposite instead? Integer format saved memory in general,but the conversion takes time.

    file_ee['a'] = file_ee['a'].astype(int)
    file_ee['b'] = file_ee['b'].astype(int)

    file_ee_routes = pd.merge(file_ee,
                       lookup_check,
                       on=['a','b'])
    #TODO: The link-based merge is performed to check if a route is interesting the study area.
    # However,the merge is slow but it takes less time than other alternatives I tested as we have to check a and b together.
    # I would appreciate some review of the following though and alternative methods.

    file_ee_routes = file_ee_routes[file_ee_routes['check']==1]
    file_ee_routes = file_ee_routes[['o','d','route','check']].drop_duplicates()
    
    file_ee_index = file_ee.index
    
    file_ee_check = pd.merge(file_ee,
                             file_ee_routes,
                             on=['o','d','route'],
                             how='left')
    del file_ee
    file_ee_check.index = file_ee_index
    file_ee = file_ee_check
    del file_ee_index, file_ee_check

    file_ee = file_ee[file_ee['check']==1]
    file_ee = file_ee.drop(columns=['check'])
    del file_ee_routes

    # TODO: same as above, but these II, EI, IE datasets are much smaller so it's not tragic.
    #  EE takes most of the work.

    file_ii = file_ii.reset_index(level=['a','b','route'])
    file_ie = file_ie.reset_index(level=['a','b','route'])
    file_ei = file_ei.reset_index(level=['a','b','route'])
    file_ii['a'] = file_ii['a'].astype(int)
    file_ie['a'] = file_ie['a'].astype(int)
    file_ei['a'] = file_ei['a'].astype(int)
    file_ii['b'] = file_ii['b'].astype(int)
    file_ie['b'] = file_ie['b'].astype(int)
    file_ei['b'] = file_ei['b'].astype(int)

    # Merge link-based information to track which route links are within the study area
    # and merge link-based distance

    #TODO: out of simplicity I am merging an integered distance and I am not considering the proportional factor from caf.space.
    # Proportions from caf.space would duplicate links, making the dataframes even bigger.
    file_ii = merge_dataframes(file_ii, lookup_check, ['a','b'], 'left')
    file_ie = merge_dataframes(file_ie, lookup_check, ['a','b'], 'left')
    file_ei = merge_dataframes(file_ei, lookup_check, ['a','b'], 'left')
    file_ee = merge_dataframes(file_ee, lookup_check, ['a','b'], 'left')

    file_ii = merge_dataframes(file_ii, link_distance, ['a','b'], 'left')
    file_ie = merge_dataframes(file_ie, link_distance, ['a','b'], 'left')
    file_ei = merge_dataframes(file_ei, link_distance, ['a','b'], 'left')
    file_ee = merge_dataframes(file_ee, link_distance, ['a','b'], 'left')

    return file_ii, file_ie, file_ei, file_ee, file_demand


def process_demand_extraction(route_support: pd.DataFrame,
                              file_demand: pd.DataFrame,
                              tld_extraction: bool):

    print(rf"Work on OD route extraction.")
    #TODO: merging link-based information to check if the link within the selected routes are internal to the study area.
    # Again a bit slow! Chunking? Any optimisation on memory or time?

    route_support = route_support.drop(columns=['a','b'])
    route_support_internal = route_support[route_support['check']==1]
    internal_distance = route_support_internal.groupby(['o','d','route'])['distance'].sum()
    total_distance = route_support.groupby(['o','d','route'])['distance'].sum()

    travelled_distance = pd.merge(internal_distance,
                                  total_distance,
                                  left_index=True,
                                  right_index=True)

    travelled_demand = pd.merge(travelled_distance,
                                file_demand,
                                left_index=True,
                                right_index=True)

    del travelled_distance

    travelled_demand = travelled_demand.set_axis(['distance_i','distance_t','pcu'], axis=1, copy=False)
    travelled_demand['pcukms_i'] = travelled_demand['pcu']*travelled_demand['distance_i']/1000
    travelled_demand['pcukms_t'] = travelled_demand['pcu']*travelled_demand['distance_t']/1000
    travelled_demand_grouped = travelled_demand.groupby(level=['o','d'])[['pcu','pcukms_i','pcukms_t']].sum()
    print(travelled_demand_grouped.pcu.sum())

    if tld_extraction == True:
        print(rf"Work on TLDs.")

        tld_support_internal = route_support_internal.groupby(['o','d','route'])['distance'].sum()
        tld_support_total = route_support.groupby(['o','d','route'])['distance'].sum()

        tld_support = pd.merge(tld_support_internal,
                               tld_support_total,
                               left_index=True,
                               right_index=True)

        # Convert kms in miles and assign distance band
        tld_support['distance_x'] = tld_support['distance_x']*MS_TO_MILES
        tld_support['distance_y'] = tld_support['distance_y']*MS_TO_MILES

        assign_cut_range(tld_support, 'distance_x', 'cut_i')
        assign_cut_range(tld_support, 'distance_y', 'cut_t')

        tld_support = tld_support.drop(columns=['distance_x','distance_y'])
        tld_support = pd.merge(tld_support,
                               file_demand,
                               left_index=True,
                               right_index=True)

        tld_i = tld_support.groupby(['cut_i'])['abs_demand'].sum().reset_index()
        tld_t = tld_support.groupby(['cut_t'])['abs_demand'].sum().reset_index()

        tld = pd.merge(TLD_RANGES[['distance_range']],
                       tld_i,
                       left_on=['distance_range'],
                       right_on=['cut_i'],
                       how='left')

        tld = pd.merge(tld,
                       tld_t,
                       left_on=['distance_range'],
                       right_on=['cut_t'],
                       how='left')

        tld = tld.drop(columns=['cut_i','cut_t'])
        tld = tld.set_axis(['distance_range','pcu_i','pcu_t'], axis=1, copy=False)

        tld = calculate_percentage(tld, 'pcu_i', [], 'pcu_i%')
        tld = calculate_percentage(tld, 'pcu_t', [], 'pcu_t%')
        tld['distance_range'] = tld['distance_range'].astype(int)
        tld.sum()


        return travelled_demand_grouped, tld

    else:
        return travelled_demand_grouped







