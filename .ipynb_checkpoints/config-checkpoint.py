import os

basedir = os.path.abspath(os.path.dirname(__file__))
class Config:

    # INPUT_FILE_PATH = os.path.join(basedir, './input')
    # INPUT_FILE_NAME = 'test_input_v1.xlsx'
    # INPUT_TAB_NAME = 'Submission'

    EXCLUDED_COLUMN = 'Exclusion'
    EXCLUDED_REASON_COLUMN = 'AdditionalInformation'
    
    MISSING_VALUE_IMPUTE = '<Missing>'
    
    DEPENDENT_COLUMN = 'Turnover'
    DEPENDENT_COLUMN_VALUES = ['Yes', 'No']

    COLUMN_TRANSFORMS = {

        "dates_to": {
            'Tenure in Company': {
                
                'date_from': 'Hire Date', 
                'date_to': 'CURRENT', 
                'timedelta': 'days',
                'discount': 0,
                'ceiling': 40
                }
        },

        "calculate_span" : {
            'Span' : {
                'manager_id': 'Manager ID',
                'employee_id': 'ID'
            }
        }
    }

    @staticmethod
    def __init__():
        pass