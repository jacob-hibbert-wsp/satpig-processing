# -*- coding: utf-8 -*-
"""
File for loading in satpig files and any other necessary files.
"""
# Built-Ins
import pandas as pd 
import re

def add_unique_id(df, id_column_name, columns_to_groupby=None):

    df.reset_index(drop=True, inplace=True)  # Reset the index in place
    """
    Assigns a unique ID to each row in the DataFrame based on the unique combinations of values in specified columns.

    Parameters:
    - df (DataFrame): The DataFrame to which unique IDs will be added.
    - id_column_name (str): The name of the column to store the unique IDs.
    - columns_to_groupby (list, optional): A list of column names whose unique combinations of values will be used to generate the IDs.
      If None, all columns except the ID column will be used. Default is None.

    Returns:
    - df (DataFrame): The DataFrame with unique IDs added.
    """
    if columns_to_groupby is None:
            df[id_column_name] = df.index + 1  # Assign unique IDs starting from 1
    else:
        df[id_column_name] = df.groupby(columns_to_groupby).ngroup() + 1
    return df
def convert_string(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
def remove_extra_commas(input_string):
    pattern = r',{2,}'  # Pattern to match two or more consecutive commas
    
    result = re.sub(pattern, ',', input_string)

    return result.rstrip(',')
def split_string_columns(df, column_name, n, cols: list):
    # Split the strings in the specified column
    df = df[column_name].str.split(',',n=7,expand=True)
    df.columns=cols
    df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric, downcast='integer')
    df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric, downcast='float')
    return df
def make_links(df_int):
    df_int['Nodes'] = df_int['Nodes'].str.split(',')
    print('split nodes')
    df_int= df_int.explode('Nodes')
    print('exploded nodes')
    df_int['b'] = df_int.groupby('route')['Nodes'].shift(-1)
    # Drop rows with NaN values in column 'b'
    df_int = df_int.dropna(subset=['b'])
    df_int = df_int.rename(columns={'Nodes': 'a'})
    df_int['link_order_id'] = df_int.groupby('route').cumcount() + 1
    df_int = add_unique_id(df_int, 'link_id', ['a','b'])
    print('link id generated')
    df_int = df_int.apply(pd.to_numeric, downcast='integer')
    print('downcasted')
    return df_int  

def read_satpig(path_to_satpig, include_connectors: bool = True):
    """
    Return a 'complete' satpig dataframe in long format.
    This can be translated/manipulated into different formats.
    """
    cols = ['o', 'd', 'uc', 'route', 'abs_demand', 'pct_demand', 'n_node', 'Nodes']
    df = pd.read_csv(path_to_satpig, sep=';', names=['dummy'])
    df = split_string_columns(df, 'dummy', 7, cols)
    df = add_unique_id(df, 'route')
    df['Nodes'] = df['Nodes'].apply(remove_extra_commas)
    df['total_links'] = df['n_node']-1
    return(df)

#if __name__=="__main__":
    #df, uc = read_satpig(r"E:\satpit_data\2018\RotherhamBase_i8c_2018_TS1_v107_SatPig_uc5.csv")
    # df = pd.read_hdf(r"E:\misc_scripts\format_tests\satpig_comp.h5", key='test', mode='r')
    #df.to_hdf(r"E:\satpit_data\2018\RotherhamBase_i8c_2018_TS1_v107_SatPig_uc5.h5", key='test', mode='w', complevel=1)
    #print('debugging')

print(8)