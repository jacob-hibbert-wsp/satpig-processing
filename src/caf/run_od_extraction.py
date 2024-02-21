# -*- coding: utf-8 -*-
"""
Created on: 2/5/2024
Updated on:

Original author: Matteo Gravellu
Last update made by:
Other updates made by:

File purpose:

"""
import os
from src.caf.OD_extraction import demand_extraction_path as de_path
from src.caf.OD_extraction import demand_extraction as extraction


# # # CONSTANTS # # #
SATPIG_PATH = r"G:\raw_data\4019 - road OD flows\Satpig\QCR"
YEAR = r"2018"

CAF_SPACE_PATH = r"G:\raw_data\caf.space\noham_link_to_lta"
ZONING_1 = r"noham"
ZONING_2 = r"lta"

P1XDUMP_PATH = r"G:\raw_data\4027 - speed & network delay\p1xdump\QCR"


# # # VARIABLES # # #
STUDY_AREA = r"Cheshire East UA" # extract this from the lookup used in the process (LTA, LAD, etc.)

USER_CLASS_LIST = ['uc1',
                   'uc2',
                   'uc3',
                   'uc4',
                   'uc5',
                   ]

TIME_PERIOD_LIST = ['TS1',
                   'TS2',
                   'TS3',
                   ]

# # # DATA REQUEST CONSTANTS # # #
OUTPUT_PATH = r"G:\data_requests\2024-01-19 CEC\output\staging"
REQUEST_CODE = r"4019 - OD"

def main(tp: str,
         uc: str):

    # Check existance of output path
    output_final_path = os.path.join(OUTPUT_PATH,
                                     REQUEST_CODE,
                                     YEAR)
    de_path.od_file_finder(output_final_path).check_output_path()

    od_flow_file = os.path.join(output_final_path,
                                rf"pcu_pcukms_{YEAR}_{tp}_{uc}.csv")

    tld_file = os.path.join(output_final_path,
                            rf"tld_{YEAR}_{tp}_{uc}.csv")


    if os.path.exists(tld_file):
        print(rf"Output for {YEAR}, {tp}, {uc} are already produced!")
    else:
        # Prepare master folders for the inputs
        if YEAR == "2018":
            satpig_path = os.path.join(SATPIG_PATH, YEAR)
            p1xdump_path = os.path.join(P1XDUMP_PATH, YEAR)

        else:
            satpig_path = os.path.join(SATPIG_PATH, YEAR, "Core")
            p1xdump_path = os.path.join(P1XDUMP_PATH, YEAR, "Core")

        caf_space_path = CAF_SPACE_PATH


        # Get the input file paths
        satpig_file = de_path.od_file_finder(satpig_path).find_satpig(tp,uc)[0]
        satpig_file_path = os.path.join(satpig_path, satpig_file)

        caf_space_file = de_path.od_file_finder(caf_space_path).find_caf_lookup(YEAR, ZONING_1, ZONING_2)[0]
        caf_space_file_path = os.path.join(caf_space_path, caf_space_file)

        p1xdump_file = de_path.od_file_finder(p1xdump_path).find_p1xdump(YEAR, tp)[0]
        p1xdump_file_path = os.path.join(p1xdump_path, p1xdump_file)

        # Run the function for the extraction
        route_support, file_demand = extraction.routing_pattern(satpig_file_path,
                                                                caf_space_file_path,
                                                                p1xdump_file_path,
                                                                STUDY_AREA)

        od_flow, demand_tld = extraction.process_demand_extraction(route_support,
                                                                   file_demand,
                                                                   True)
        del route_support, file_demand

        # Save outputs
        od_flow.to_csv(od_flow_file, index=True)
        demand_tld.to_csv(tld_file, index=False)
        print(rf"Outputs are saved for {YEAR}, {tp}, {uc}")
        del od_flow, demand_tld


if __name__ == "__main__":
    for tp in TIME_PERIOD_LIST:
        for uc in USER_CLASS_LIST:
            main(tp, uc)

















