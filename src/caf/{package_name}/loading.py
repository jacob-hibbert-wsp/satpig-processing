# -*- coding: utf-8 -*-
"""
File for loading in satpig files and any other necessary files.
"""
# Built-Ins
import pandas as pd
import dask.dataframe as dd
from copy import deepcopy
# Third Party
from caf.toolkit.concurrency import multiprocess
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #
# # # FUNCTIONS # # #
def convert_string(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
def split_string_columns(df, column_name, n, cols: list):
    # Split the strings in the specified column
    df[column_name] = df[column_name].str.split(',')

    # Separate into two columns: the first n elements and the remaining elements
    df['FirstNElements'] = df[column_name].apply(lambda x: [convert_string(val) for val in x[:n]])
    df['Nodes'] = df[column_name].apply(lambda x: ','.join(x[n:]))

    # Drop the original column with strings if needed

    df[[cols[i] for i in range(n)]] = pd.DataFrame(df['FirstNElements'].tolist(),
                                                                    index=df.index)
    df.drop(columns=[column_name, 'FirstNElements'], inplace=True)
    return df

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
    links.set_index(['n_link', 'o', 'd', 'route', 'order', 'a', 'b'], inplace=True)
    return links

def read_satpig(path_to_satpig, include_connectors: bool = True):
    """
    Return a 'complete' satpig dataframe in long format.
    This can be translated/manipulated into different formats.
    """
    cols = ['o', 'd', 'uc', 'route', 'abs_demand', 'pct_demand', 'n_node']
    df = pd.read_csv(path_to_satpig, sep=';', names=['dummy'])
    df = split_string_columns(df, 'dummy', 7, cols)
    uc = df['uc'].unique()[0]
    df.drop('uc', axis=1, inplace=True)
    df.set_index(['o', 'd', 'route', 'abs_demand', 'pct_demand'], inplace=True)
    dfs = []
    for i in df['n_node'].unique():
        dfs.append((df[df['n_node'] == i].copy(), i))
    del(df)
    df_out = multiprocess(internal_multi, arg_list=dfs)
    return pd.concat(df_out), uc

if __name__=="__main__":
    df, uc = read_satpig(r"Y:\Carbon\QCR_Assignments\07.Noham_to_NoCarb\2018\RotherhamBase_i8c_2018_TS2_v107_SatPig_uc4.csv")
    # df = pd.read_hdf(r"E:\misc_scripts\format_tests\satpig_comp.h5", key='test', mode='r')
    df.to_hdf(r"E:\misc_scripts\format_tests\satpig_comp.h5", key='test', mode='w', complevel=1)
    print('debugging')