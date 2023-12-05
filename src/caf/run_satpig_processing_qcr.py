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
HOME_FOLDER = r"G:\raw_data\4019 - road OD flows\Satpig\QCR"

YEAR_LIST = ['2028',
             #'2038',
             #'2043',
             #'2048',
             ]
USERCLASS_LIST = ['1',
                  '2',
                  '3',
                  '4',
                  '5',
                  ]
SCENARIO_LIST = ['Core',
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
    df, uc = loading.read_satpig(satpig_path)

    # save processed satpig
    processed_satpig_path = os.path.join(satpig_folder, rf"{satpig_file}.h5")
    df.to_hdf(processed_satpig_path, key='test', mode='w', complevel=1)

    print(rf"Satpig processing completed for {year}, {scenario}, {tp}, {uc}!")

def main():
    for year in YEAR_LIST:
        for scenario in SCENARIO_LIST:
            for tp in TP_LIST:
                for uc in USERCLASS_LIST:
                    call_satpig_processing(HOME_FOLDER,
                                           year,
                                           scenario,
                                           tp,
                                           uc)

if __name__=="__main__":
    main()