import pandas as pd

#look output from caf.space scratch_6 in line_to_poly branch
#coulmns are link a node, links b node, zone link segment is in, how much of the link is in that zone
Lookup = pd.read_csv(r"C:\Users\JacobHibbert\OneDrive - Transport for the North\Documents\misc_coding\line_to_poly_trans\outputs\noham2028_msoa2011_2028.csv")
links = pd.read_csv(r'A:\QCR- assignments\03.Assignments\h5files\2028\2028_link_table.csv')
########### making link only have one zone ##############################

#when loadig in links: groupby a and b, get the first
links = links.groupby(["a", "b"], as_index=False)[["speed", "distance"]].first()
Lookup.rename(columns={'msoa2011_id':'zone', 'A':'a', 'B':'b'}, inplace=True)
print(Lookup)
Lookup=Lookup.loc[Lookup.groupby(['a', 'b'])['factor'].idxmax()]
print(Lookup)
links = pd.merge(Lookup,links, on = ['a', 'b'], validate="1:1", how="left")
print(links['zone'])
links['speed'] =links['speed'].fillna(48.0)
links['distance'] = links['distance'].fillna(1)
links = links[['a','b','zone', 'speed', 'distance']]

############### merging converting zones to integers ############################################
#lad is a lookup between zones of interest and LAD areas
lad = pd.read_csv(r"A:\QCR- assignments\03.Assignments\h5files\Other Inputs\MSOA11_WD21_LAD21_EW_LU_1.csv")
#links = pd.read_csv(r"A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\2018_link_table_new.csv")
print(links)
# Merge the DataFrames and bring the new zone into links table
merged_df = pd.merge(links, lad[['zone_cd', 'zone']], validate = 'm:1', left_on='zone', right_on='zone_cd', how='left')
 #Drop the redundant 'zone_cd' column and rename the 'zone' column from lad
merged_df = merged_df.drop(columns=['zone_cd', 'zone_x']).rename(columns={'zone_y': 'zone'}).dropna()
merged_df['zone']= merged_df['zone'].astype(int)
print(merged_df)

merged_df.to_csv(r'A:\QCR- assignments\03.Assignments\h5files\2028\2028_link_table_fixed.csv', index=False)

dups = merged_df.duplicated(["a", "b"]).sum()
if dups > 0:
    raise ValueError(f"found {dups} duplicate link IDs")