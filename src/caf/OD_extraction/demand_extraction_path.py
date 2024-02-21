# -*- coding: utf-8 -*-
"""
Created on: 5/18/2023
Updated on:

Original author: Matteo Gravellu
Last update made by:
Other updates made by:

File purpose:

"""
# Built-Ins
import enum
import logging
import os
# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position
# Local imports here
# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
LOG = logging.getLogger(__name__)


# # # CLASSES # # #
class od_file_finder:

    def __init__(self, directory):
        self.directory = directory

    def find_satpig(self,
                    time_period,
                    user_class,
                    ):
        """
        Find CSV file(s) containing any one of the specified substrings in their filenames in the given directory.

        Parameters:
            time_period (str): List of substrings to search for in the filenames.
            user_class (str): Format of the file to look for in the directory.

        Returns:
            list: List of filenames containing any one of the specified substrings.
        """
        # List all files in the directory
        files = os.listdir(self.directory)

        # Filter CSV files containing any one of the specified substrings
        matching_files = [file for file in files if
                          time_period in file and
                          user_class in file and
                          file.endswith('.h5')]

        return matching_files

    def find_caf_lookup(self,
                        year,
                        zoning_1,
                        zoning_2):
        """
        Find CSV file(s) containing any one of the specified substrings in their filenames in the given directory.

        Parameters:
            year (str): List of substrings to search for in the filenames.
            zoning_1 (list): List of substrings to search for in the filenames.
            zoning_2 (str): Format of the file to look for in the directory.

        Returns:
            list: List of filenames containing any one of the specified substrings.
        """
        # List all files in the directory
        files = os.listdir(self.directory)

        # Filter CSV files containing any one of the specified substrings
        matching_files = [file for file in files if
                          year in file and
                          zoning_1 in file and
                          zoning_2 in file and
                          file.endswith('.csv')]

        return matching_files


    def find_noham_lookup(self,
                        zoning_1,
                        zoning_2):
        """
        Find CSV file(s) containing any one of the specified substrings in their filenames in the given directory.

        Parameters:
            year (str): List of substrings to search for in the filenames.
            zoning_1 (list): List of substrings to search for in the filenames.
            zoning_2 (str): Format of the file to look for in the directory.

        Returns:
            list: List of filenames containing any one of the specified substrings.
        """
        # List all files in the directory
        files = os.listdir(self.directory)

        # Filter CSV files containing any one of the specified substrings
        matching_files = [file for file in files if
                          zoning_1 in file and
                          zoning_2 in file and
                          file.endswith('.csv')]

        return matching_files

    def find_p1xdump(self,
                     year,
                     time_period,
                     ):
        """
        Find CSV file(s) containing any one of the specified substrings in their filenames in the given directory.

        Parameters:
            year (str): List of substrings to search for in the filenames.
            time_period (str): List of substrings to search for in the filenames.

        Returns:
            list: List of filenames containing any one of the specified substrings.
        """
        # List all files in the directory
        files = os.listdir(self.directory)

        # Filter CSV files containing any one of the specified substrings
        matching_files = [file for file in files if
                          year in file and
                          time_period in file and
                          file.endswith('.csv')]

        return matching_files


    def check_output_path(self):

        output_path = self.directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        else:
            pass


# DIRECTORY = r"G:\raw_data\4019 - road OD flows\Satpig\QCR\2018"
# SUBSTRINGS1_LIST = 'TS3'
# SUBSTRINGS2_LIST = 'uc5'
#
# folder = od_file_finder(DIRECTORY)
# matching_files = folder.find_satpig(SUBSTRINGS1_LIST,SUBSTRINGS2_LIST)
#
# for file in matching_files:
#     print(file)