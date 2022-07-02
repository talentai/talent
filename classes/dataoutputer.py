
import pandas as pd
from io import BytesIO


import logging
logger = logging.getLogger('DataOutputer')

class DataOutputer():
    
    def __init__(self, config, dataLoader):

        self.reverseFieldTypeDict = self.reverse_dictionary(dataLoader.fieldTypeDict)
        self.excludedColumn = config['EXCLUDED_COLUMN']
        self.excludedReasonColumn = config['EXCLUDED_REASON_COLUMN']

    @staticmethod
    def reverse_dictionary(dict):
        
        dict_reverse = {}
        for key, value in dict.items():
            for string in value:
                dict_reverse.setdefault(string, key)
        
        return dict_reverse

    def output_validation_data(self, data):

        column_reorder = ([self.excludedColumn, self.excludedReasonColumn] + 
                          [col for col in data.columns if col not in [self.excludedColumn, self.excludedReasonColumn]])

        data = data[column_reorder]
        data.columns = pd.MultiIndex.from_arrays((data.columns, data.columns.map(self.reverseFieldTypeDict)))
        data = data.swaplevel(1, 0, axis=1)

        return data
    
    def get_excel_file_object(self, data):
        
        output = BytesIO()
        
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        data.to_excel(writer, index=True, sheet_name='Submission')
        workbook = writer.book
        worksheet = writer.sheets['Submission']
        
        # Expand the column width for a better visual
        # for column in data:
        #         column_width = max(data[column].astype(str).map(len).max(), len(column))+3
        #         col_idx = data.columns.get_loc(column)
        #         writer.sheets['Submission'].set_column(col_idx, col_idx, column_width)
        # cell_format = workbook.add_format()  
        
        writer.save()
        
        processed_data = output.getvalue()

        return processed_data

