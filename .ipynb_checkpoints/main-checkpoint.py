from datetime import datetime
import numpy as np
import pandas as pd

from config import Config
from classes import DataLoader, DataValidator, Transformer, DataOutputer

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

config = dict(Config.__dict__)

df = pd.read_excel(r'input/test_input_v2.xlsx', 'Submission', header=[0, 1]) 

data_loader = DataLoader(df)
data_validator = DataValidator(config, data_loader)

validation_results, data = data_validator.apply_validation()

data_outputer = DataOutputer(config, data_loader)
validation_output = data_outputer.output_validation_data(data)
excel_object = data_outputer.get_excel_file_object(validation_output)

transformer = Transformer(config)
data = transformer.apply_transformations(data)

validation_output.to_csv('data.csv')