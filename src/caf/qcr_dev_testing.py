import pandas as pd 
import numpy as np
import h5py
import sys
from caf.toolkit.concurrency import multiprocess
filename = r"C:\Users\JacobHibbert\Downloads\NoHAM_QCR_DM_High_2038_TS1_v107_SatPig_uc1.csv"
#with h5py.File(r"C:\Users\JacobHibbert\Downloads\testing.h5",'w') as h5f:
#
#    i_arr=np.arange(10)
#    x_arr=np.arange(10.0)
#
#    my_dt = np.dtype([ ('i_arr', int), ('x_arr', float) ] )
#    table_arr = np.recarray( (10,), dtype=my_dt )
#    table_arr['i_arr'] = i_arr
#    table_arr['x_arr'] = x_arr
#
#    my_ds = h5f.create_dataset('/ds1',data=table_arr)
#
## read 1 column using numpy slicing: 
#with h5py.File(r"C:\Users\JacobHibbert\Downloads\testing.h5",'r') as h5f:
#
#    h5_i_arr = h5f['ds1'][:,'i_arr']
#    h5_x_arr = h5f['ds1'][:,'x_arr']
#    print (h5_i_arr)
#    print (h5_x_arr)
#sys.exit()
#import tables
#h5file = tables.open_file(r"G:\raw_data\4001, 4008, 4019, 4026 - Highway OD flows\raw_data\Satpig\QCR\2038\High\NoHAM_QCR_DM_High_2038_TS1_v107_SatPig_uc1.h5", driver="H5FD_CORE")
#print(h5file)
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
    df = df[column_name].str.split(',',n=7,expand=True)
    df.columns=cols
    df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric, downcast='integer')
    df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric, downcast='float')

    df['route'] = np.arange(df.shape[0])#df['o'].astype(str)+'_'+df['d'].astype(str)+'_'+df['route'].astype(str)
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
    uc = df['uc'].unique()[0]
    h5File = r"C:\Users\JacobHibbert\Downloads\NoHAM_QCR_DM_High_2038_TS1_v107_SatPig_uc1_test.h5"
    #df.to_hdf(h5File, key="/data/d1",format = 'table',data_columns = ['o','d','route','uc','abs_demand', 'pct_demand'], errors='ignore', index = False)
    df= multiprocess(internal_multi, arg_list=df)
    return df

df = read_satpig(filename)
print(df)
print(df.dtypes)

# Export the pandas DataFrame into HDF5

#h5File = r"C:\Users\JacobHibbert\Downloads\NoHAM_QCR_DM_High_2038_TS1_v107_SatPig_uc1_test.h5"

#df.to_hdf(h5File, "/data/d1")

# Use pandas again to read data from the hdf5 file to the pandas DataFrame

#df1 = pd.read_hdf(h5File, "/data/d1")

#print("DataFrame read from the HDF5 file through pandas:")

#print(df1)