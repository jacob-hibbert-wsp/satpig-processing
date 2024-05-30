import pandas as pd


Lookup = pd.read_csv(r"C:\Users\JacobHibbert\OneDrive - Transport for the North\Documents\misc_coding\line_to_poly_trans\outputs\noham2018_msoa2011_2.csv")
links = pd.read_csv(r'A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\2018_link_table.csv')
########### making link only have one zone ##############################

#when loadig in links: groupby a and b, get the first for 
links = links.groupby(["a", "b"], as_index=False)[["speed", "distance"]].first()
Lookup.rename(columns={'msoa2011_id':'zone', 'A':'a', 'B':'b'}, inplace=True)
print(Lookup)
Lookup=Lookup.loc[Lookup.groupby(['a', 'b'])['factor'].idxmax()]
print(Lookup)
links = pd.merge(Lookup,links, on = ['a', 'b'], validate="1:1")
print(links)
links = links[['a','b','zone', 'speed', 'distance']]

############### merging converting zones to integers ############################################

lad = pd.read_csv(r"A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\MSOA11_WD21_LAD21_EW_LU_1.csv")
#links = pd.read_csv(r"A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\2018_link_table_new.csv")
print(links)
# Merge the DataFrames and bring the new zone into links table
merged_df = pd.merge(links, lad[['zone_cd', 'zone']], validate = 'm:1', left_on='zone', right_on='zone_cd', how='left')
 #Drop the redundant 'zone_cd' column and rename the 'zone' column from lad
merged_df = merged_df.drop(columns=['zone_cd', 'zone_x']).rename(columns={'zone_y': 'zone'}).dropna()
merged_df['zone']= merged_df['zone'].astype(int)
print(merged_df)

merged_df.to_csv(r'A:\QCR- assignments\03.Assignments\h5files\BaseYearFiles\2018_link_table_new_2.csv', index=False)

dups = merged_df.duplicated(["a", "b"]).sum()
if dups > 0:
    raise ValueError(f"found {dups} duplicate link IDs")