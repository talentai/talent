from datetime import datetime
import re
import pandas as pd

import logging
logger = logging.getLogger('DataValidator')

class DataValidator():
    
    def __init__(self, config, dataLoader):

        self.data = dataLoader.data
        self.fieldTypeDict = dataLoader.fieldTypeDict
        self.excludedColumn = config['EXCLUDED_COLUMN']
        self.excludedReasonColumn = config['EXCLUDED_REASON_COLUMN']
        self.missingValueImpute = config['MISSING_VALUE_IMPUTE']  
        self.dependentColumn = config['DEPENDENT_COLUMN'] 
        self.dependentColumnValues = config['DEPENDENT_COLUMN_VALUES'] 

    def apply_validation(self):
        '''
        This is the main method of DataValidator, which controls the flow.
        '''

        self.data[self.excludedReasonColumn] = ''

        summary_results = {}

        summary_results['unknown_values_dependent_column'] = self.check_dependent_column_values()
        summary_results['numeric_columns_with_invalid_dtype'] = self.check_numeric_datatypes()
        if 'date' in list(self.fieldTypeDict.keys()):
            summary_results['date_columns_with_invalid_values'] = self.check_date_values()
        
        self.data[self.excludedReasonColumn] = self.data[self.excludedReasonColumn].str[:-3] # clean up the end of the text
        self.data[self.excludedColumn] = False
        self.data.loc[self.data[self.excludedReasonColumn]!='', self.excludedColumn] = True

        return self.prepare_warning_message(summary_results), self.data


    def check_numeric_datatypes(self):
        logger.debug('Checking numeric column datatypes')
        
        invalid_datatype_numeric_columns = []

        for numeric_column in self.fieldTypeDict['numeric']:

            # Check if any values are missing
            self.data[numeric_column].fillna(self.missingValueImpute, inplace=True)

            if self.data[numeric_column].dtype not in ['int64', 'int', 'float']:
                self.data[numeric_column] = self.data[numeric_column].apply(lambda x: str(x).replace('$','').replace(',',''))
              
                try:
                    self.data[numeric_column] = self.data[numeric_column].astype('float')
                except:
                    invalid_datatype_numeric_columns.append(numeric_column)
                    non_numeric_rows = self.data[numeric_column].apply(self.detect_non_digit_char).values
                    self.data.loc[non_numeric_rows, self.excludedReasonColumn] += 'Non-numeric value found in numeric column [{}] | '.format(numeric_column)
                    
                    # Overwrite the value to the col minimum, and cast it back to numeric
                    self.data.loc[non_numeric_rows, numeric_column] = 0
                    self.data[numeric_column] = self.data[numeric_column].astype('float')

        return invalid_datatype_numeric_columns
    
    @staticmethod
    def detect_non_digit_char(x):
        
        # Find any character other than 0-9 digit and signle dot: [^0-9\.]
        # Or anywhere has 2 dots: \.\.
        if re.findall('^(\d+(\.\d+)?)$', str(x).strip()) != []:
            return False
        else:
            return True

    def check_date_values(self):

        invalid_date_columns = []

        for date_column in self.fieldTypeDict['date']:
            self.data[date_column].fillna(self.missingValueImpute, inplace=True)
            dates = self.data[date_column].copy()
            
            dates_converted = pd.to_datetime(dates, errors='coerce')
            invalid_dates = dates_converted.isnull()

            if invalid_dates.sum():
                invalid_date_columns.append(date_column)

            self.data.loc[invalid_dates, self.excludedReasonColumn] += 'Value [' + self.data.loc[invalid_dates, date_column] + '] in date column [' + date_column + '] is not valid (date overriden) | '
            self. data[date_column] = dates_converted.fillna(pd.to_datetime('2000-01-01'))
        
        return invalid_date_columns

    @staticmethod
    def check_category_column_values(column, allowable_values):
        
        values_in_data = column.unique().tolist()
        unknown_values = list(set(values_in_data) - set(allowable_values))
    
        return unknown_values

    def check_dependent_column_values(self):
        logger.debug('Checking for unknown categorical values in dependent variable')

        self.data[self.dependentColumn].fillna(self.missingValueImpute, inplace=True)

        if self.data[self.dependentColumn].dtype != object:
            self.data[self.dependentColumn] = self.data[self.dependentColumn].astype(str)

        unknown_values = self.check_category_column_values(self.data[self.dependentColumn], self.dependentColumnValues)
        
        for unknown_value in unknown_values:
            self.data.loc[(self.data[self.dependentColumn]==unknown_value), self.excludedReasonColumn] += 'Invalid value [{}] for column [{}] | '.format(unknown_value, self.dependentColumn)

        return unknown_values

    def prepare_warning_message(self, summary_results):
        warning_message = []

        if summary_results['numeric_columns_with_invalid_dtype']:
            warning_message.append('Some rows contained invalid numeric values.')
        if 'date' in list(self.fieldTypeDict.keys()) and summary_results['date_columns_with_invalid_values']:
            warning_message.append('Some rows contained invalid date values.')
        if summary_results['unknown_values_dependent_column']:
            warning_message.append('Some rows contained invalid values in dependent column.')
        if warning_message:
            return ' '.join(warning_message)
        else:
            return None

    def split_excluded(self):
        excluded = self.data[(self.data[self.excluded_column]==True)].copy()
        self.data = self.data[(self.data[self.excludedColumn]==False)].copy()

        if len(self.data)==0:
            logger.error('No rows left for analysis. All excluded. First excluded reason:\n{}'.format(excluded.head()['AdditionalInformation'].values[0]))
            raise Exception('No rows left for analysis. All excluded')
        
        return self.data, excluded