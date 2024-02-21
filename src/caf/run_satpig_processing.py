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
print(os.getcwd())
# Local Imports
from routing import loading

# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
HOME_FOLDER_2018 = r"Y:\Carbon\QCR_Assignments\07.Noham_to_NoCarb\2018"
BASE_NAME_2018_AM = r"RotherhamBase_i8c_2018_TS1_v107_SatPig_uc"
BASE_NAME_2018_IP = r"RotherhamBase_i8c_2018_TS2_v107_SatPig_uc"
BASE_NAME_2018_PM = r"RotherhamBase_i8c_2018_TS3_v107_SatPig_uc"


SCENARIO_LIST = {
    #"am_2018": [HOME_FOLDER_2018, BASE_NAME_2018_AM],
    "ip_2018": [HOME_FOLDER_2018, BASE_NAME_2018_IP],
    "pm_2018": [HOME_FOLDER_2018, BASE_NAME_2018_PM],
}

USERCLASS_LIST = [1,
                  2,
                  3,
                  4,
                  5]
# # # CLASSES # # #

# # # FUNCTIONS # # #
def call_satpig_processing(home_folder: str,
                          satpig_name: str,
                          userclass: int,
                          ) -> None:

    # read satpig extract
    satpig_path = os.path.join(home_folder, rf"{satpig_name}{userclass}.csv")
    df, uc = loading.read_satpig(satpig_path)
    print(rf"Satpig processing initiated for: {satpig_path} - userclass:{userclass}")

    # save processed satpig
    processed_satpig_path = os.path.join(home_folder, rf"{satpig_name}{userclass}.h5")
    df.to_hdf(processed_satpig_path, key='test', mode='w', complevel=1)

    print(rf"Satpig processing completed!")

def main():
    for scenario in SCENARIO_LIST:
        home_folder = SCENARIO_LIST[scenario][0]
        satpig_name = SCENARIO_LIST[scenario][1]

        for uc in USERCLASS_LIST:
            call_satpig_processing(home_folder,
                                  satpig_name,
                                  uc)

if __name__=="__main__":
    main()