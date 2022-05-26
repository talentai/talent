import numpy as np
import pandas as pd

import logging
logger = logging.getLogger('DataLoader')

class DataLoader():

    def __init__(self, data):

        self.read_data_and_type(data)


    def read_data_and_type(self, data):

        self.data = data.copy()

        self.extract_field_type_dictionary()
        self.extract_clean_data()
        self.extract_column_name()


    def extract_field_type_dictionary(self):

        self.data.dropna(axis=1, how='all', inplace=True)
        
        self.fieldTypeDict = {}

        for type, field in list(self.data.columns):
            self.fieldTypeDict.setdefault(type, []).append(field)


    def extract_clean_data(self):
        
        self.data.columns = self.data.columns.get_level_values(1)

    def extract_column_name(self):
        
        self.column_name = self.data.columns.tolist()