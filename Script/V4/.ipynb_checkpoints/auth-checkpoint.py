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
import io

import locale
import requests

import pyrebase as pb

# from PE_Functions import *
# from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

from datetime import datetime

# Firebase Authentication-----------------------------------------------------------------------------------------------------------------------
firebaseConfig = {
    'apiKey': "AIzaSyC6yIfxcoRtWmNUQZDER3ZZPgXf1ZbEcTw",
    'authDomain': "talent-turnover.firebaseapp.com",
    'projectId': "talent-turnover",
    'storageBucket': "talent-turnover.appspot.com",
    'messagingSenderId': "1008257138200",
    'appId': "1:1008257138200:web:645e7b9138bcfb8659cd6d",
    'measurementId': "G-X3MFWDFWTY",
    'databaseURL': "https://talent-turnover-default-rtdb.firebaseio.com/"  
}

fb = pb.initialize_app(firebaseConfig)
auth = fb.auth()
db = fb.database()
storage = fb.storage()

# Helper Functions -----------------------------------------------------------------------------------------------------------------------
def get_excel_file_downloader_html(data, file_label='File'):
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:file/xlsx;base64,{bin_str}" download="{file_label}">{file_label}</a>'
    return href

def get_excel_file_downloader_url(data, file_label='File'):
    bin_str = base64.b64encode(data).decode()
    href = f'"data:file/xlsx;base64,{bin_str}"'
    return href

def clear_state():
    for key in st.session_state.keys():
        del st.session_state[key]

def clear_file(exclude_list):
    for key in st.session_state.keys():
        if key not in exclude_list:
            del st.session_state[key]

# Streamlit initialize session state to track login, upload status, data, and others ------------------------------------------------ 
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = 'No'

if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'upload_status' not in st.session_state:
    st.session_state['upload_status'] = "No"
        
if 'username' not in st.session_state:
    st.session_state['username'] = ""
    
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Streamlit Login UI -----------------------------------------------------------------------------------------------------------------------
st.sidebar.title("Please login to start:")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'],index=0,on_change=clear_state)

# Obtain User Input for email and password
login_place = st.sidebar.empty()
login_container = login_place.container()

# App 
# Sign up Block
if choice == 'Sign up':
    username = st.sidebar.text_input(
        'Please input your user name', value='Default')
    company = st.sidebar.text_input(
        'Please input your company name', value='Default')
    submit = st.sidebar.button('Create my account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account is created suceesfully!')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("ID").set(user['localId'])
        db.child(user['localId']).child("Username").set(username)
        db.child(user['localId']).child("Company").set(company)
        db.child(user['localId']).child("Email").set(email)
        db.child(user['localId']).child("Password").set(password)
        
        st.title('Welcome' + username)
        st.info('Login via login drop down selection')
        st.session_state['login_status'] = 'Yes'
        st.session_state['user'] = user

# Login Block

if (choice == 'Login') and (st.session_state['login_status'] == "No"):
    email = login_container.text_input('Please enter your email address')
    password = login_container.text_input('Please enter your password',type = 'password')
    login = login_container.button('Login')
    if login:
        try:
            user = auth.sign_in_with_email_and_password(email,password)
            # user_info = auth.get_account_info()
            username = db.child(user['localId']).child("Username").get().val()
            st.session_state['login_status'] = "Yes"
            st.session_state['user'] = user
            st.session_state['username'] = username
            login_place.empty()
        except:
            st.write('User not found, please sign up with drop down selection')
            st.stop()
           
        st.title('Welcome')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# if st.session_state['login_status'] == 'Yes':
#     username = st.session_state['username']
#     st.sidebar.write('Welcome ' + username)
#     logout = st.sidebar.button('Logout',on_click=clear_state,key = 'logout_continue')
#     if logout:
#         # login_place.empty()
#         st.experimental_rerun()
        
if st.session_state['login_status'] == 'Yes':
    st.write(st.session_state)
    user = st.session_state['user']
    username = st.session_state['username']
    st.sidebar.write('Welcome ' + username)
    logout = st.sidebar.button('Logout',on_click=clear_state,key = 'logout_continue')
    if logout:
        # login_place.empty()
        st.experimental_rerun()
        
    bio = st.radio('Jump to',['Home','Calculation'])

    if bio == 'Calculation':
        cal_start = st.number_input('Insert a number')
        st.write('your starting input is: '+str(cal_start))

        cal_add = st.number_input('Add a number')
        st.write('your add number is: '+ str(cal_add))

        cal_result = cal_start+cal_add
        st.write('your final number is: '+str(cal_result))

        butt_save = st.button('Save result')
        if butt_save:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
            cal_save = {'Calculation' : cal_result,
                    'Timestamp' : dt_string} 

            db.child(user['localId']).child("Calculation").push(cal_save)
            st.balloons()
        st.write(st.session_state)
        
    elif bio == 'Home':
        st.write("You are home!")
        exclude_list = ['login_status','user','username','data']
        uploaded_file = st.file_uploader('Step 1: Upload Data Template', type=['xlsx'], on_change=clear_file,args=[exclude_list])
        df = None
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file,sheet_name="Sheet1")
            df['price_update'] = df['price']*100
            st.session_state.data = df
            st.session_state.upload_status = "Yes"
        elif st.session_state.upload_status=="Yes":
            df = st.session_state.data
            
        if df is not None:
            st.write(df['price'][0])
            st.write(df['price_update'][0])
        
        df_name = st.text_input('Filename', '')
        butt_save_data = st.button('Save data')
        
        if butt_save_data:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
            data_save = {'Filename':df_name,
                    'File' : df.to_dict(),
                    'Timestamp' : dt_string} 
            db.child(user['localId']).child("Data").push(data_save)
            
            # output = BytesIO()
            # writer = pd.ExcelWriter(output, engine='xlsxwriter')
            # df.to_excel(writer, index=False, sheet_name='Sheet1')
            # workbook = writer.book
            # worksheet = writer.sheets['Sheet1']
            # writer.save()
            # processed_data = output.getvalue()
            
            # save_path = get_excel_file_downloader_url(processed_data, 'save_data.xlsx')
            # save_path = 'Test/test.xlsx'
            
            save_path = uploaded_file
            
            # uid = user['localId']
            fireb_upload = storage.child(user['localId']).put(save_path,user['idToken'])
            data_url = storage.child(user['localId']).get_url(fireb_upload['downloadTokens']) 
            db.child(user['localId']).child("Data").child("Store_URL").push(data_url)

            st.write(save_path)
            st.balloons()
        
        butt_load_data = st.button('Load data')
        butt_load_file = st.button('Load file')
        
        if butt_load_data:
            data_all = db.child(user['localId']).child("Data").get().val()
            if data_all is not None:
                val = db.child(user['localId']).child("Data").get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                val_by_time = db.child(user['localId']).sort(val, "Timestamp")
                for myfile in val_by_time.each():
                    myfile = myfile.val()
                    if myfile['Filename'] == 'first':
                        download = pd.DataFrame.from_dict(myfile['File']).to_excel('download.xlsx')
                    st.write(myfile['Filename'])
                    st.write(myfile['Timestamp'])
        
        if butt_load_file:
            file_all = db.child(user['localId']).child("Data").child("Store_URL").get().val()
            if file_all is not None:
                val = db.child(user['localId']).child("Data").child("Store_URL").get()
                mydata = val.each()[0]
                # load_data = mydata.val()+'?raw=true'
                load_data = mydata.val()
                st.write(load_data)
                
                # get_content = requests.get(load_data).content
                # read_data = pd.read_excel(io.StringIO(get_content.decode('utf-8')))
                # read_data = pd.read_csv(get_content, encoding= 'unicode_escape')
                read_data = pd.read_excel(load_data,engine='openpyxl')
                # read_data = pd.read_excel(load_data)
                st.write(read_data)
                
                # get_content = requests.get(load_data).content
                # r = requests.get(load_data)
                # open('temp.xls', 'wb').write(r.content)
                # read_data = pd.read_excel('temp.xls')
                
                read_data = pd.read_excel(load_data,engine='openpyxl')
                # read_data['price_update10'] = read_data['price'] * 10
                
                # st.write(read_data)
#                 for mydata in val.each():
#                     load_data = mydata.val()
#                     read_data = pd.read_excel(load_data)
                    
#                     read_data['price_update2'] = read_data['price_update'] * 10
                    # st.write(read_data.head(1))
                    
                    # if myfile['Filename'] == 'first':
                    #     download = pd.DataFrame.from_dict(myfile['File']).to_excel('download.xlsx')
                    # st.write(load_data)
                    # st.write(read_data)
                    # st.write(myfile['Timestamp'])
        
        st.write(st.session_state)
