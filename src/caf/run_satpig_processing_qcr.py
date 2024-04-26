# -*- coding: utf-8 -*-
"""
Created on: 11/23/2023
Updated on:

Original author: Matteo Gravellu
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins

# Third Party
import os
import sys
#print(os.getcwd())
# Local Imports
from routing import loading

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
HOME_FOLDER = r"G:\raw_data\4001, 4008, 4019, 4026 - Highway OD flows\raw_data\Satpig\QCR"

YEAR_LIST = [#'2028',
             '2038',
             #'2043',
             #'2048',
             ]
USERCLASS_LIST = ['1',
                  '2',
                  '3',
                  '4',
                  '5',
                  ]
SCENARIO_LIST = [#'Core',
                 'High',
                 'Low',
                 ]
TP_LIST = ['TS1',
           'TS2',
           'TS3',
           ]

# # # CLASSES # # #

# # # FUNCTIONS # # #
def call_satpig_processing(home_folder: str,
                           year: str,
                           scenario: str,
                           tp: str,
                           uc: str,
                           ) -> None:
 
    # read satpig extract
    satpig_file = rf"NoHAM_QCR_DM_{scenario}_{year}_{tp}_v107_SatPig_uc{uc}"
    satpig_folder = os.path.join(home_folder, year, scenario)
    satpig_path = os.path.join(satpig_folder, rf"{satpig_file}.csv")

    print(rf"Satpig processing initiated for: {satpig_file}")
    df = loading.read_satpig(satpig_path)
    df.set_index(['o','d','route','uc', 'total_links'], inplace=True)

    # save processed satpig
    processed_satpig_path = os.path.join(r"C:\Users\Ferrari\JacobHibbert_Secondment\satpit_output", rf"{satpig_file}.h5")
    df[['abs_demand', 'pct_demand']].to_hdf(processed_satpig_path,key="/data/OD",format = 'fixed', complevel=1)
    print('OD Done')
    df = df.reset_index()
    df.drop(['abs_demand', 'pct_demand', 'n_node','o', 'd', 'uc','total_links'], axis=1, inplace=True)
    df = loading.make_links(df)
    print(df[['route', 'link_id', 'link_order_id']])
    df.set_index(['route','link_id'],inplace = True)
    df[['link_order_id']].to_hdf(processed_satpig_path, key="/data/Route",format = 'fixed', complevel=1)
    print('Route Done')
    print(df.columns)
    df = df.reset_index()
    df.drop(['route', 'link_order_id'],axis=1, inplace=True)
    print(df)
    df = df[['link_id','a','b']].drop_duplicates(subset=['link_id', 'a', 'b'])
    df = df.reset_index()
    df = df.sort_values(by='link_id')
    print(df)
    df.set_index(['link_id'],inplace = True)
    df[['a','b']].to_hdf(processed_satpig_path, key="/data/link",format = 'fixed', complevel=1)#,data_columns = ['link_id','a','b'], errors='ignore', index = False)
    print('link done')
    print(rf"Satpig processing completed for {year}, {scenario}, {tp}, {uc}!")

def main():
    for year in YEAR_LIST:
        for scenario in SCENARIO_LIST:
            for tp in TP_LIST:
                for uc in USERCLASS_LIST:
                    try:
                        call_satpig_processing(HOME_FOLDER,
                                           year,
                                           scenario,
                                           tp,
                                           uc)
                    except Exception:
                        print(rf"Could not produce: {year}, {scenario}, {tp}, {uc}!")

if __name__=="__main__":
    main()