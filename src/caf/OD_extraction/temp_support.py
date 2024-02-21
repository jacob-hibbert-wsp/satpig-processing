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
import dask.dataframe as dd
import datetime
# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
FOLDER = r"G:\raw_data\4019 - road OD flows\Satpig\QCR\2018"
FILE = r"RotherhamBase_i8c_2018_TS1_v107_SatPig_uc1.h5"

CAF_FOLDER = r"G:\raw_data\caf.space"
CAF_FILE = r"noham2018_lta.csv"

P1XDUMP_FOLDER = r"G:\raw_data\p1xdump\QCR\2018"
P1XDUMP_FILE = r"link_table_NoHAM_Base_2018_TS1_v106_I6.csv"

NOHAM_LTA_LOOKUP = r"G:\raw_data\caf.space\noham_lta_spatial.csv"

TEMPLATE_OUTPUT = r"G:\raw_data\6001 - under processing\templates"

# TLD CONSTANTS
TLD_RANGES = pd.DataFrame({
    'distance_range': ['1','2', '3', '5', '10', '15', '25', '35', '50', '100', '200', '200+'],
    'min_distance': [0, 1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200],
    'max_distance': [1, 2, 3, 5, 10, 15, 25, 35, 50, 100, 200, 100000000]
})


DIST_FACTOR = 1609.344


# # # CLASSES # # #

# # # FUNCTIONS # # #
def assign_distance_range(distance):
    for index, row in TLD_RANGES.iterrows():
        if row['min_distance'] <= distance <= row['max_distance']:
            return row['distance_range']
    return None  # Return None if no range matches


# read noham to zoning lookup
noham_lookup = pd.read_csv(NOHAM_LTA_LOOKUP)
noham_internal = noham_lookup[noham_lookup['lta_id']=='West Yorkshire Combi']
noham_internal = noham_lookup['noham_id'].tolist()


# read satpig routing input
file_path = os.path.join(FOLDER, FILE)
file = pd.read_hdf(file_path, key='test', mode='r')
file = file.drop(columns=['pct_demand'])

# extract the o-d-route based demand and keep it as a separate dataframe
file_demand = file.droplevel(['n_link','order','a','b'])
file_demand = file_demand[~file_demand.index.duplicated(keep='first')]
#file_demand = file.reset_index(level=['o','d','route'])


# get rid of the demand and proceed with routing analysis
file = file.reset_index(level=['a','b'])
file = file.drop(columns=['abs_demand'])
file['a'] = file['a'].astype(int)
file['b'] = file['b'].astype(int)
file.dtypes

# read lookup
caf_file_path = os.path.join(CAF_FOLDER, CAF_FILE)
lookup = pd.read_csv(caf_file_path)
lookup = lookup.set_axis(['a','b','area','factor'], axis=1, copy=False)
lookup_area = lookup[lookup['area']=='West Yorkshire Combi']
lookup_check = lookup_area[['a','b']]
lookup_check['check']=1

#o = lookup_area['a'].to_list()
#d = lookup_area['b'].to_list()

# read p1xdump file
distance_file_path = os.path.join(P1XDUMP_FOLDER, P1XDUMP_FILE)
link_distance = pd.read_csv(distance_file_path, usecols=['A','B','Distance'])
link_distance = link_distance.set_axis(['a','b','distance'], axis=1, copy=False)
link_distance['distance'] = link_distance['distance'].astype(int)


# Merge routing and area lookup to check if the links are in the study area
# and filter routes if they have at least one link in the study area
#file_ab_index = file.index
file_ab = pd.merge(file,
                   lookup_check,
                   on=['a','b'],
                   how='left')
file_ab.index = file.index
file_ab['check'] = file_ab['check'].fillna(0)
del file

# Keep routes that are flagged for the study area
#route_area = file_ab[file_ab['check']==1]
#route_area = route_area.drop(columns=['a','b'])
#route_area = route_area.droplevel(['n_link','order'])
#route_area = route_area[~route_area.index.duplicated(keep='first')]
#route_area = route_area.reset_index(level=['o','d','route'])
#route_area = route_area.set_axis(['o','d','route','r_check'], axis=1, copy=False)

#route_area_sum = route_area.groupby(['o','d','route'])['r_check'].sum()
#route_area.sum()


# In the original routing dataframe, merge the route_area information
# to check if the route is part of the study area and drop routes that are not part of it
file_ab = file_ab.reset_index(level=['o','d','route'])
route_area_sum = file_ab.groupby(['o','d','route'])['check'].sum().reset_index()


print(route_area_sum.head(9))
file_index = file_ab.index

file_ab = pd.merge(file_ab,
                   route_area_sum,
                   on=['o','d','route'],
                   how='left')

file_ab.index = file_index
del file_index, route_area_sum
file_ab.dtypes

file_ab = file_ab[file_ab['check_y']>0]

file_ab = file_ab.reset_index(level=['n_link','order'])
file_ab = file_ab.rename(columns={'check_x': 'check_link', 'check_y': 'check_route'})
print(file_ab.head(10))

# The routes that are within the area share the same number of links (n_links)
# and total number of links within the area (sum)
route_ii = file_ab[file_ab['n_link']==file_ab['check_route']]
route_notii = file_ab[file_ab['n_link']!=file_ab['check_route']]
del file_ab

route_ii = route_ii.drop(columns=['check_route'])
route_notii = route_notii.drop(columns=['check_route'])
print(route_notii.head(8))

# Filter the not_within df into first and last links and apply the following conditions:
# If the first link is within the study area, assign support == 1 (pattern: from)
# If the last link is within the study area, assign support == 2 (pattern: to)
route_notii['support']=0

route_notii_first = route_notii[route_notii['order']==1]
route_notii_last = route_notii[route_notii['order']==route_notii['n_link']]

route_notii_first['support'] = route_notii_first['check_link']
route_notii_last['support'] = route_notii_last['check_link']*2


route_notii_firstlast = pd.concat([route_notii_first,route_notii_last])
print(route_notii_firstlast.head(8))
del route_notii_first, route_notii_last

route_notii_firstlast = route_notii_firstlast.groupby(['o','d','route'])['support'].sum()
route_notii_firstlast = route_notii_firstlast.reset_index()


route_notii = route_notii.drop(columns=['support'])

# Merge back support information on the first and last route links
# to the not_within routing dataframe

route_notii_support = pd.merge(route_notii,
                               route_notii_firstlast,
                               on=['o','d','route'],
                               how='left')

del route_notii, route_notii_firstlast
print(route_notii_support.head(16))



# Use sum support to understand route pattern for not_within routes:
# 0: through, 1: from, 2: to, 3: from-out-to
# and merge it back to the not within file route dataframe
route_notii_support = route_notii_support.drop(columns=['order','n_link'])
route_notii_support = pd.merge(route_notii_support,
                               link_distance,
                               on=['a','b'],
                               how='left')
route_notii_support['distance'] = route_notii_support['distance'].fillna(0)


# quickly recover the internal routes to attach them to the non internal dataframe
route_ii = route_ii.drop(columns=['n_link','order'])
route_ii['support']=4

route_ii_support = pd.merge(route_ii,
                            link_distance,
                            on=['a', 'b'],
                            how='left')

route_support = pd.concat([route_ii_support, route_notii_support], axis=0)
del route_ii_support, route_notii_support
print(route_support.columns)



# Attach distance-based information and separate not_within dataframes
# following the rule-based segmenat
routes_through = file_route_check_notwithin_support[file_route_check_notwithin_support['support_y']==0]
routes_from = file_route_check_notwithin_support[file_route_check_notwithin_support['support_y']==1]
routes_to = file_route_check_notwithin_support[file_route_check_notwithin_support['support_y']==2]
routes_from_out_to = file_route_check_notwithin_support[file_route_check_notwithin_support['support_y']==3]
del file_route_check_notwithin_support


routes_through = routes_through.drop(columns=['support_y','a','b'])
routes_from = routes_from.drop(columns=['support_y','a','b'])
routes_to = routes_to.drop(columns=['support_y','a','b'])
routes_from_out_to = routes_from_out_to.drop(columns=['support_y','a','b'])

routes_through['check'] = routes_through['check'].fillna(0)
routes_from['check'] = routes_from['check'].fillna(0)
routes_to['check'] = routes_to['check'].fillna(0)
routes_from_out_to['check'] = routes_from_out_to['check'].fillna(0)

routes_through_groupby = routes_through.groupby(['o','d','route','abs_demand','check'])['distance'].sum()
routes_from_groupby = routes_from.groupby(['o','d','route','abs_demand','check'])['distance'].sum()
routes_to_groupby = routes_to.groupby(['o','d','route','abs_demand','check'])['distance'].sum()
routes_from_out_to_groupby = routes_from_out_to.groupby(['o','d','route','abs_demand','check'])['distance'].sum()

routes_through_groupby_total = routes_through.groupby(['o','d','route','abs_demand'])['distance'].sum()
routes_from_groupby_total = routes_from.groupby(['o','d','route','abs_demand'])['distance'].sum()
routes_to_groupby_total = routes_to.groupby(['o','d','route','abs_demand'])['distance'].sum()
routes_from_out_to_groupby_total = routes_from_out_to.groupby(['o','d','route','abs_demand'])['distance'].sum()

print(routes_through_groupby.head(16))
print(routes_through_groupby_total.head(16))

del routes_through, routes_from, routes_to, routes_from_out_to

routes_through_groupby = routes_through_groupby.reset_index()
routes_from_groupby = routes_from_groupby.reset_index()
routes_to_groupby = routes_to_groupby.reset_index()
routes_from_out_to_groupby = routes_from_out_to_groupby.reset_index()

routes_through_groupby_total = routes_through_groupby_total.reset_index()
routes_from_groupby_total = routes_from_groupby_total.reset_index()
routes_to_groupby_total = routes_to_groupby_total.reset_index()
routes_from_out_to_groupby_total = routes_from_out_to_groupby_total.reset_index()

routes_through = pd.merge(routes_through_groupby,
                          routes_through_groupby_total,
                          on=['o','d','route','abs_demand'],
                          how='left')


routes_from = pd.merge(routes_from_groupby,
                          routes_from_groupby_total,
                          on=['o','d','route','abs_demand'],
                          how='left')


routes_to = pd.merge(routes_to_groupby,
                          routes_to_groupby_total,
                          on=['o','d','route','abs_demand'],
                          how='left')


routes_from_out_to = pd.merge(routes_from_out_to_groupby,
                          routes_from_out_to_groupby_total,
                          on=['o','d','route','abs_demand'],
                          how='left')

routes_through_na = routes_through[routes_through.isna().any(axis=1)]
routes_from_na = routes_from[routes_from.isna().any(axis=1)]
routes_to_na = routes_to[routes_to.isna().any(axis=1)]
routes_from_out_to_na = routes_from_out_to[routes_from_out_to.isna().any(axis=1)]

del routes_through_groupby, routes_from_groupby, routes_to_groupby, routes_from_out_to_groupby
del routes_through_groupby_total, routes_from_groupby_total, routes_to_groupby_total, routes_from_out_to_groupby_total


routes_through = routes_through[routes_through['check']==1]
routes_from = routes_from[routes_from['check']==1]
routes_to = routes_to[routes_to['check']==1]
routes_from_out_to = routes_from_out_to[routes_from_out_to['check']==1]

routes_through['pcu_kms_i'] = routes_through['abs_demand']*routes_through['distance_x']
routes_through['pcu_kms_tot'] = routes_through['abs_demand']*routes_through['distance_y']
routes_from['pcu_kms_i'] = routes_from['abs_demand']*routes_from['distance_x']
routes_from['pcu_kms_tot'] = routes_from['abs_demand']*routes_from['distance_y']
routes_to['pcu_kms_i'] = routes_to['abs_demand']*routes_to['distance_x']
routes_to['pcu_kms_tot'] = routes_to['abs_demand']*routes_to['distance_y']
routes_from_out_to['pcu_kms_i'] = routes_from_out_to['abs_demand']*routes_from_out_to['distance_x']
routes_from_out_to['pcu_kms_tot'] = routes_from_out_to['abs_demand']*routes_from_out_to['distance_y']

os.chdir(TEMPLATE_OUTPUT)
demand_pcukms_through = routes_through.groupby(['o','d'])[['abs_demand','pcu_kms_i','pcu_kms_tot']].sum().reset_index()
demand_pcukms_from = routes_from.groupby(['o','d'])[['abs_demand','pcu_kms_i','pcu_kms_tot']].sum().reset_index()
demand_pcukms_to = routes_to.groupby(['o','d'])[['abs_demand','pcu_kms_i','pcu_kms_tot']].sum().reset_index()
demand_pcukms_from_out_to = routes_from_out_to.groupby(['o','d'])[['abs_demand','pcu_kms_i','pcu_kms_tot']].sum().reset_index()

demand_pcukms_through.to_csv(r"demand_pcukms_ee.csv", index=False)
demand_pcukms_from.to_csv(r"demand_pcukms_ie.csv", index=False)
demand_pcukms_to.to_csv(r"demand_pcukms_ei.csv", index=False)
demand_pcukms_from_out_to.to_csv(r"demand_pcukms_iei.csv", index=False)



# convert meters in miles for the TLD extracts and assign TLD range
routes_through['distance_x'] = routes_through['distance_x']/DIST_FACTOR
routes_through['distance_y'] = routes_through['distance_y']/DIST_FACTOR
routes_through['tld_i'] = routes_through['distance_x'].apply(assign_distance_range)
routes_through['tld_tot'] = routes_through['distance_y'].apply(assign_distance_range)

routes_from_out_to['distance_x'] = routes_from_out_to['distance_x']/DIST_FACTOR
routes_from_out_to['distance_y'] = routes_from_out_to['distance_y']/DIST_FACTOR
routes_from_out_to['tld_i'] = routes_from_out_to['distance_x'].apply(assign_distance_range)
routes_from_out_to['tld_tot'] = routes_from_out_to['distance_y'].apply(assign_distance_range)

routes_from['distance_x'] = routes_from['distance_x']/DIST_FACTOR
routes_from['distance_y'] = routes_from['distance_y']/DIST_FACTOR
routes_from['tld_i'] = routes_from['distance_x'].apply(assign_distance_range)
routes_from['tld_tot'] = routes_from['distance_y'].apply(assign_distance_range)

routes_to['distance_x'] = routes_to['distance_x']/DIST_FACTOR
routes_to['distance_y'] = routes_to['distance_y']/DIST_FACTOR
routes_to['tld_i'] = routes_to['distance_x'].apply(assign_distance_range)
routes_to['tld_tot'] = routes_to['distance_y'].apply(assign_distance_range)

# aggregate TLDs (internal + total) and concat into single dataframe
routes_through_tld_i = routes_through.groupby(['tld_i'])['abs_demand'].sum().reset_index()
routes_from_tld_i = routes_from.groupby(['tld_i'])['abs_demand'].sum().reset_index()
routes_to_tld_i = routes_to.groupby(['tld_i'])['abs_demand'].sum().reset_index()
routes_from_out_to_tld_i = routes_from_out_to.groupby(['tld_i'])['abs_demand'].sum().reset_index()

routes_through_tld_tot = routes_through.groupby(['tld_tot'])['abs_demand'].sum().reset_index()
routes_from_tld_tot = routes_from.groupby(['tld_tot'])['abs_demand'].sum().reset_index()
routes_to_tld_tot = routes_to.groupby(['tld_tot'])['abs_demand'].sum().reset_index()
routes_from_out_to_tld_tot = routes_from_out_to.groupby(['tld_tot'])['abs_demand'].sum().reset_index()

routes_through_tld = pd.merge(TLD_RANGES[['distance_range']],routes_through_tld_i,left_on=['distance_range'],right_on=['tld_i'], how='left')
routes_through_tld = pd.merge(routes_through_tld,routes_through_tld_tot,left_on=['distance_range'],right_on=['tld_tot'], how='left')
routes_through_tld = routes_through_tld.drop(columns=['tld_i','tld_tot'])
routes_through_tld = routes_through_tld.set_axis(['distance_range','tld_i','tld_tot'], axis=1, copy=False).set_index('distance_range')
routes_through_tld_sum = routes_through_tld['tld_i'].sum()
routes_through_tld_perc = routes_through_tld/routes_through_tld_sum*100
routes_through_tld_perc.sum()

routes_from_tld = pd.merge(TLD_RANGES[['distance_range']],routes_from_tld_i,left_on=['distance_range'],right_on=['tld_i'], how='left')
routes_from_tld = pd.merge(routes_from_tld,routes_from_tld_tot,left_on=['distance_range'],right_on=['tld_tot'], how='left')
routes_from_tld = routes_from_tld.drop(columns=['tld_i','tld_tot'])
routes_from_tld = routes_from_tld.set_axis(['distance_range','tld_i','tld_tot'], axis=1, copy=False).set_index('distance_range')
routes_from_tld_sum = routes_from_tld['tld_i'].sum()
routes_from_tld_perc = routes_from_tld/routes_from_tld_sum*100
routes_from_tld_perc.sum()

routes_to_tld = pd.merge(TLD_RANGES[['distance_range']],routes_to_tld_i,left_on=['distance_range'],right_on=['tld_i'], how='left')
routes_to_tld = pd.merge(routes_to_tld,routes_to_tld_tot,left_on=['distance_range'],right_on=['tld_tot'], how='left')
routes_to_tld = routes_to_tld.drop(columns=['tld_i','tld_tot'])
routes_to_tld = routes_to_tld.set_axis(['distance_range','tld_i','tld_tot'], axis=1, copy=False).set_index('distance_range')
routes_to_tld_sum = routes_to_tld['tld_i'].sum()
routes_to_tld_perc = routes_to_tld/routes_to_tld_sum*100
routes_to_tld_perc.sum()

routes_from_out_to_tld = pd.merge(TLD_RANGES[['distance_range']],routes_from_out_to_tld_i,left_on=['distance_range'],right_on=['tld_i'], how='left')
routes_from_out_to_tld = pd.merge(routes_from_out_to_tld,routes_from_out_to_tld_tot,left_on=['distance_range'],right_on=['tld_tot'], how='left')
routes_from_out_to_tld = routes_from_out_to_tld.drop(columns=['tld_i','tld_tot'])
routes_from_out_to_tld = routes_from_out_to_tld.set_axis(['distance_range','tld_i','tld_tot'], axis=1, copy=False).set_index('distance_range')
routes_from_out_to_tld_sum = routes_from_out_to_tld['tld_i'].sum()
routes_from_out_to_tld_perc = routes_from_out_to_tld/routes_from_out_to_tld_sum*100
routes_from_out_to_tld_perc.sum()

routes_tld_perc = pd.concat([routes_through_tld_perc, routes_from_tld_perc, routes_to_tld_perc, routes_from_out_to_tld_perc],
                            keys=['EE','IE','EI','IEI'])

routes_tld_perc.to_csv("notwithin_tld.csv")



### Process within routes










