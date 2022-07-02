from datetime import datetime
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger('Transformer')

class Transformer:
    
    def __init__(self, config):
        
        self.columnTransforms = config['COLUMN_TRANSFORMS']

    def apply_transformations(self, data):
        '''The main method of this class'''

        self.data = data.copy()

        self.apply_dates_to()
        self.calculate_span()
        
        return self.data

    def apply_dates_to(self):
        logger.info('Preparing dates...')
        # args: (end, start, subtract, discount, floor, ceiling)
        if len(self.columnTransforms['dates_to']) > 0:
            for column, kwargs in self.columnTransforms['dates_to'].items():
                logger.debug('Creating data column %s', column)
                #start_year[start_year>end_year]-= 100   # fix for assuming 20XX rather than 19XX
                self.data[column] = self.years_delta_from_dates(self.data, **kwargs)

    @staticmethod
    def years_delta_from_dates(data, date_from, date_to, timedelta='years', discount=0, floor=0, ceiling=np.inf):

        if date_to == 'CURRENT':
            end = datetime.now()
        else:
            end = pd.to_datetime(data[date_to])

        start = pd.to_datetime(data[date_from])

        if timedelta=='years':
            end = end.dt.year.values if end else datetime.now().year # add var to control this, dont want this all the time
            start = end.dt.year.values
            date_delta = (end - start)
        # elif timedelta=='months':
        elif timedelta=='days':
            date_delta = (end - start).apply(lambda x: x.days)/365.25

        date_delta[date_delta < 0]+= 100 # cover scenario if start is later then end
        discounted_delta = date_delta - discount
        # Can use np.clip(x, min, max)
        return  (
            discounted_delta
            .apply(lambda x: max(x, floor))
            .apply(lambda x: min(x, ceiling))
        )
    @staticmethod
    def force_string_data_types(data, column):

        # Force EmpID to string
        # Note this weird hack replacing '.0' with ''.
        # Basically if either of these come in looking numberic, but there is a nan, then the 'str' turns them
        # to float 123.0 - which ruins span calculation and add's mess. So do this to prevent it...
        data[column] = data[column].apply(lambda x: str(x).rstrip('0').rstrip('.') if '.0' == str(x)[-2:] else str(x))

    def calculate_span(self):

        if len(self.columnTransforms['calculate_span']) > 0:
            logger.info('Calculating span...')

        for column, values in self.columnTransforms['calculate_span'].items():

            logger.debug('Creating data column %s', column)

            managerID = values['manager_id']
            employeeID = values['employee_id']

            self.force_string_data_types(self.data, managerID)
            self.force_string_data_types(self.data, employeeID)

            span_df = (
                self.data
                .groupby(managerID)
                .size().reset_index()
            )
            
            span_df.columns = [managerID, column]

            self.data = self.data.merge(
                 span_df,
                 how='left',
                 left_on=employeeID,
                 right_on=managerID,
                 suffixes=('', '_merged')
                 )
            
            self.data.drop(managerID + '_merged', axis=1, inplace=True)
            self.data[column] = self.data[column].fillna(0)
