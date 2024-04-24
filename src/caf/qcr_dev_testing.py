import pandas as pd 
import numpy as np
import h5py
import sys
from caf.toolkit.concurrency import multiprocess
import re
filename = r"C:\Users\JacobHibbert\Downloads\NoHAM_QCR_DM_Core_2038_TS1_v107_SatPig_uc2.csv"

def add_unique_id(df, id_column_name):
    """
    Assigns a unique ID to each row in the DataFrame based on the index.

    Parameters:
    - df (DataFrame): The DataFrame to which unique IDs will be added.
    - id_column_name (str): The name of the column to store the unique IDs.

    Returns:
    - df (DataFrame): The DataFrame with unique IDs added.
    """
    df.reset_index(drop=True, inplace=True)  # Reset the index in place
    df[id_column_name] = df.index + 1  # Assign unique IDs starting from 1
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
    df = add_unique_id(df, 'route')
    return df
def make_links(df_int):
    df_int.drop(['abs_demand', 'pct_demand', 'n_node','o', 'd', 'uc','total_links'], axis=1, inplace=True)
    df_int['Nodes'] = df_int['Nodes'].str.split(',')
    df_int= df_int.explode('Nodes')
    df_int['b'] = df_int.groupby('route')['Nodes'].shift(-1)
    # Drop rows with NaN values in column 'b'
    df_int = df_int.dropna(subset=['b'])
    df_int = df_int.rename(columns={'Nodes': 'a'})
    df_int['link_order_id'] = df_int.groupby('route').cumcount() + 1
    df_int = add_unique_id(df_int, 'link_id')
    df_int = df_int.apply(pd.to_numeric, downcast='integer')
    return df_int   
def internal_multi(df_int, i):
    df_int.drop('n_node', axis=1, inplace=True)
    df_int['Nodes'] = df_int['Nodes'].str.split(',')
    df_int = df_int.join(pd.DataFrame(df_int['Nodes'].tolist(), index=df_int.index))
    drop_cols = ['Nodes', i]
    df_int.drop(columns=drop_cols, inplace=True)
    if len(df_int.columns) < 2:
        return
    a = df_int[range(0, i - 1)]
    b = df_int[range(1, i)].stack().to_frame()
    a.columns += 1
    a = a.stack().to_frame()
    a.columns = ['a']
    b.columns = ['b']
    links = a.join(b).reset_index()
    links.rename(columns={'level_5': 'order'}, inplace=True)
    links['n_link'] = i-1
    print(links.columns)
    links.set_index(['n_link', 'o', 'd', 'route', 'order', 'a', 'b'], inplace=True)
    return links

def read_satpig(path_to_satpig, include_connectors: bool = True):
    """
    Return a 'complete' satpig dataframe in long format.
    This can be translated/manipulated into different formats.
    """
    cols = ['o', 'd', 'uc', 'route', 'abs_demand', 'pct_demand', 'n_node', 'Nodes']
    df = pd.read_csv(path_to_satpig, sep=';', names=['dummy'])
    df = split_string_columns(df, 'dummy', 7, cols)
    df['Nodes'] = df['Nodes'].apply(remove_extra_commas)
    df['total_links'] = df['n_node']-1
    #uc = df['uc'].unique()[0]
    h5File = r"C:\Users\JacobHibbert\Downloads\NoHAM_QCR_DM_Core_2038_TS1_v107_SatPig_uc2_test.h5"
    df.to_hdf(h5File, key="/data/d1",format = 'table',data_columns = ['o','d','route','uc','abs_demand', 'pct_demand', 'total_links'], errors='ignore', index = False)
    
    df = make_links(df)
    df.to_hdf(h5File, key="/data/d2",format = 'table',data_columns = ['route','link_id','link_order_id'], errors='ignore', index = False)
    df.to_hdf(h5File, key="/data/d3",format = 'table',data_columns = ['link_id','a','b'], errors='ignore', index = False)
    return df

df = read_satpig(filename)
print(df)
print(df.dtypes)