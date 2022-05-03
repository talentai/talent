import streamlit as st
import pandas as pd
import numpy as np

from patsy import dmatrices
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os
import time

import locale

import pyrebase

# from PE_Functions import *
# from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

import streamlit as st
import datetime
import random

from streamlit_option_menu import option_menu

# if ('countit' or 'upload_status' or 'change_count','data') not in st.session_state:
#     st.session_state.countit = 0
#     st.session_state.upload_status = 0
#     st.session_state.change_count = 0
#     st.session_state.data = None
    
if 'change_count' not in st.session_state:
    # st.session_state.countit = 0
    # st.session_state.upload_status = 0
    st.session_state['change_count'] = 0
    # st.session_state.data = None

if 'upload_status' not in st.session_state:
    st.session_state['upload_status'] = 0
    
if 'data' not in st.session_state:
    st.session_state['data'] = None

st.write(st.session_state)
    
def countme():
    st.session_state.count =  st.session_state.count+1

# st.write("each run time count before")
# st.write(st.session_state.countit)
# st.session_state.count =  st.session_state.count+1
# st.write("each run time count after")
# st.write(st.session_state.count)
    
def clear_state():
    for key in st.session_state.keys():
        del st.session_state[key]

menu = st.sidebar.container()

# select = st.sidebar.radio("Menu",('Home', 'Tasks'))

with menu:
    select = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="vertical")    
    
uploaded_file = st.sidebar.file_uploader('Step 1: Upload Data Template', type=['xlsx'], on_change=clear_state)
df = None

@st.experimental_memo
def data_loader(uploaded_file):
    df = None
    if uploaded_file is not None:
        # st.write('file uploaded')
        # st.session_state.countit =  st.session_state.countit+1
        if st.session_state.upload_status == 0:
            df = pd.read_excel(uploaded_file,sheet_name="Sheet1")
            df['price_update'] = df['price']*100
            st.session_state.data = df
            st.session_state.upload_status = 1 
            # st.session_state.change_count =  st.session_state.change_count+1
            time.sleep(5)
            print('sleeping '+str(random.randrange(20, 50, 3)))
    return df
    
df = data_loader(uploaded_file)

# st.session_state.change_count =  st.session_state.change_count+1

# select = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")

# st.write(st.session_state)
# st.write(selected2)
# st.write(st.session_state.count)

home = st.empty()
task = st.empty()

homepage = st.container()
taskpage = st.container()

if select == 'Home':
    homepage.write('home')
    st.session_state['change_count'] =  st.session_state['change_count']+1
    # st.write(st.session_state)
    # task.empty()
#     homepage.write(select)
#     homepage.info('correct in home page')
    if df is not None:
# #         homepage.write(st.session_state.data['price'][0])
# #         homepage.write(st.session_state.data['price_update'][0])
# #         homepage.write(st.session_state)
        homepage.write(df['price'][0])
        homepage.write(df['price_update'][0])
        homepage.write(st.session_state)

elif select == 'Tasks':
    taskpage.write('tasks')
    st.session_state['change_count'] =  st.session_state['change_count']+1
    # st.write(st.session_state)
    
    # home.empty()
    # taskpage.write(select)
    # taskpage.info('correct in task page')
    if df is not None:
    #     # taskpage.write(st.session_state.data['price'][1])
    #     # taskpage.write(st.session_state.data['price_update'][1])
#     # taskpage.write(st.session_state)
        taskpage.write(df['price'][1])
        taskpage.write(df['price_update'][1])
        taskpage.write(st.session_state)
        
# st.title('Counter Example')
# st.upload

# if ('count' or 'happy') not in st.session_state:
#     st.session_state.count = 0
#     st.session_state.happy = 10

# print(dict(inc=2,times=3))
    
# def add_num(inc,times):
#     st.session_state.count = st.session_state.count+inc*times
#     st.session_state.happy = st.session_state.happy + st.session_state.count

# def reset_all():
#     for key in st.session_state.keys():
#         del st.session_state[key]

# increment = st.button('Increment',on_click=add_num, kwargs=dict(inc=2,times=3))
# reset = st.button('Reset',on_click=reset_all)

# st.write('Count = ', st.session_state.count)
# st.write(st.session_state)

# # st.write('Count = ', st.session_state.count)

# st.check

# st.title('Counter Example')
# if 'count' not in st.session_state:
#     st.session_state.count = 0
#     st.session_state.last_updated = datetime.time(0,0)

# def update_counter():
#     st.session_state.count += st.session_state.increment_value
#     st.session_state.last_updated = st.session_state.update_time

# with st.form(key='my_form'):
#     st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
#     st.number_input('Enter a value', value=0, step=1, key='increment_value')
#     submit = st.form_submit_button(label='Update', on_click=update_counter)

# st.write('Current Count = ', st.session_state.count)
# st.write('Last Updated = ', st.session_state.last_updated)

# if "celsius" not in st.session_state:
#     # set the initial default value of the slider widget
#     st.session_state.celsius = 50.0

# st.slider(
#     "Temperature in Celsius",
#     min_value=-100.0,
#     max_value=100.0,
#     key="celsius"
# )

# # This will get the value of the slider widget
# st.write(st.session_state.celsius)