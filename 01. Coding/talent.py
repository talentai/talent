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

import locale

import pyrebase

from PE_Functions import *
from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

# from streamlit_option_menu import option_menu

# Set Path
st.set_page_config(layout="wide")
# demo_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Demo.xlsx'
file_path = Path(__file__).parents[0].__str__()+'/Data/Pay Equity Data Template.xlsx'
display_path = Path(__file__).parents[0].__str__()+'/Data/Display Name.xlsx'
style_path = Path(__file__).parents[0].__str__()+'/Style/style.css'
with open(style_path) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Set Styles
# metric = st.markdown("""
#     <style>
#     div.css-12w0qpk.e1tzin5v2
#          {background-color: #EFF8F7
#          }
#     div.css-1ht1j8u.e16fv1kl0
#         {font-size: 15px; 
#         }
#     </style>""", unsafe_allow_html=True)

# info_card = st.markdown("""
#     <style>
#     div.css-21e425.e1tzin5v4 {font-size: 5px}
#     </style>""", unsafe_allow_html=True)

m = st.markdown("""
    <style>
    div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #3498DB;background-color:#3498DB;border-radius:5px;border:2px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:8px 25px;text-decoration:none;
    &:active {position:relative;top:1px;}}
    </style>""", unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Set Functions

# UI *********************************
# Setup Session State:
# if 'demo_run' not in st.session_state:
#     st.session_state['demo_run'] = 'no'
# Side Panel


# m_col1,m_col2 = st.sidebar.columns((1, 1))
# m_col1_but = m_col1.button('See Demo')
# m_col2_but = m_col2.button('Close Demo')

# st.sidebar.markdown("""---""")

# if "demo_box" not in st.session_state:
#     st.session_state.demo_box = False

st.sidebar.header(' 🎯 Start here')
demo_check = st.sidebar.checkbox('See Demo', key='demo_box')

# Step 1: Download Template
# st.sidebar.markdown("Step 1: 🖱️ 'Save link as...'")
# st.sidebar.markdown(get_binary_file_downloader_html(file_path, 'Download Instruction and Data Template'), unsafe_allow_html=True)

# Step 2: Upload File
if demo_check==False:
    st.sidebar.markdown("Step 1: 🖱️ 'Save link as...'")
    st.sidebar.markdown(get_binary_file_downloader_html(file_path, 'Download Instruction and Data Template'), unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader('Step 2: Upload Data Template', type=['xlsx'])
else:
    st.sidebar.write('Please clear the "See Demo" checkbox to start your analysis.')
    uploaded_file = None

# Step 3: Check empty columns
# st.sidebar.write('Step 3: Review the output in the main panel')
# st.sidebar.write('Step 3: Confirm Selected Configuration')
# config = st.sidebar.form("Configuration")
# with config:
#     # config.write("A. Choose fair pay confidence internal")
#     ci = config.slider(label = 'A. Choose fair pay confidence internal %', min_value = 70, max_value = 99, step = 1, help='Setting at 95% means I want to have a pay range so that 95% of the time, the predicted pay falls inside.')
#     # checkbox_val = st.checkbox("Form checkbox")
#     # Every form must have a submit button.
#     submitted_form = config.form_submit_button("🚀 Confirm to Run Analysis'")
    

# st.sidebar.write("Choose fair pay confidence internal at: "+str(ci))
# st.sidebar.write('form submit' + str(submitted_form))
# st.sidebar.write('file submit' + str(uploaded_file is not None))

# submit_butt = False
# if ((uploaded_file is not None) and (submitted_form == True)):
#     submit_butt = True

# submit_butt = False
# if ((uploaded_file is not None)):
#     submit_butt = st.sidebar.button("Submit")

# st.sidebar.write('Step 3: Review the output in the main panel')
# st.sidebar.write('If you wish to launch your data after the demonstration, please uncheck the "See Demo" box.')

st.sidebar.markdown("""---""")

# m_col1,m_col2 = st.sidebar.columns((1, 1))
# m_col1_but = m_col1.button('See Demo')
# m_col2_but = m_col2.button('Close Demo')
    
# st.sidebar.write('Final submit' + str(submit_butt))

# option_menu

# selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")

# selected2 = option_menu(None, ["Settings", "Pay Gap Result", "Fair Pay Calculator"], 
#     icons=['gear', 'cloud-upload', "list-task"], 
#     menu_icon="cast", default_index=0, orientation="horizontal")


# st.write(selected2)

# Main Panel-------------------------------------------
c1, c2 = st.columns((2, 1))

c1.title('PayX')
c1.write('PayX measure the value and the statistical significance of the **net gender/ethnicity pay gap**. That is, we compare pay between men and women, between white and black with similar level, function, location, experience and performance, etc to ensure the difference is gender/ethnicity based. Statistical significance allows us to quantify if a gap is due to chance or gender/ethnicity bias.')
c2.image('Picture/salary.jpeg',use_column_width='auto')

# st.markdown("""---""")

# with st.expander("🔔 See Instruction"):
#     st.write("""To start your analysis, please upload data in sidebar. Check out "See Demo" for a sample output.""")
    # e1, e2 = st.columns((1,4))
    # e1.image('Picture/guide2.jpeg',use_column_width='auto')
    # e2.write("""To start your analysis, please upload data in sidebar. Check out "See Demo" for a sample output.""")
    
main_page = st.container()
with main_page.container():
    main_page_info = main_page.empty()
    # st.write(submit_butt)
    if uploaded_file is not None:
        main_page_info.info('Running input file.')
        # analysis(df_submit = uploaded_file, run_demo = False, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info, ci = ci)
        analysis(df_submit = uploaded_file, run_demo = False, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
        # st.session_state["demo_box"] = False
    else:
        m_info = main_page_info.info('Awaiting the upload of the data template.')
        if demo_check:
            analysis(df_submit = None, run_demo = True, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
        # else:
        #     st.experimental_rerun()
        
#         m_col1,m_col2,t1 = main_page.columns((1, 1, 2))
        
#         m_col1_but = m_col1.button('See Demo')
#         m_col2_but = m_col2.button('Close Demo')
#         if m_col1_but:
#             # analysis(df_submit = None, run_demo = True, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info, ci = ci)
#             analysis(df_submit = None, run_demo = True, file_path = file_path, display_path = display_path, main_page = main_page, main_page_info = main_page_info)
            
#         if m_col2_but:
#             # main_page.empty()
#             st.experimental_rerun()

