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
from streamlit_option_menu import option_menu

# from PE_Functions import *
# from PE_Parameter import *

from pathlib import Path

import xlsxwriter
from io import BytesIO

from datetime import datetime

# Streamlit CSS Style Setup
st.set_page_config(layout="wide")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
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

# Firebase Authentication-----------------------------------------------------------------------------------------------------------------------
firebaseConfig = {
    'apiKey': "AIzaSyC6yIfxcoRtWmNUQZDER3ZZPgXf1ZbEcTw",
    'authDomain': "talent-turnover.firebaseapp.com",
    'projectId': "talent-turnover",
    'storageBucket': "talent-turnover.appspot.com",
    'messagingSenderId': "1008257138200",
    'appId': "1:1008257138200:web:645e7b9138bcfb8659cd6d",
    'measurementId': "G-X3MFWDFWTY",
    'databaseURL': "https://talent-turnover-default-rtdb.firebaseio.com"  
}

# st.write(st.secrets["firebase_secrets"]["databaseURL"])

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
if 'login_time' not in st.session_state:
    st.session_state['login_time'] = 0
    
if 'run_time' not in st.session_state:
    st.session_state['run_time'] = 0

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = 'No'

if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'upload_status' not in st.session_state:
    st.session_state['upload_status'] = "No"
        
if 'username' not in st.session_state:
    st.session_state['username'] = ""

if 'email' not in st.session_state:
    st.session_state['email'] = ""
    
if 'user' not in st.session_state:
    st.session_state['user'] = None
    
if 'cal_result' not in st.session_state:
    st.session_state['cal_result'] = np.nan

# st.write(st.session_state)
    
# Streamlit -----------------------------------------------------------------------------------------------------------------------

# Authentication -----------------------------------------------------------------------------------------------------------------------
choice_place = st.empty()
choice_container = choice_place.container()

# Obtain User Input for email and password
login_place = st.empty()
login_container = login_place.container()

signup_place = st.empty()
signup_container = signup_place.container()

# st.write("Run again")
# st.session_state['run_time'] = st.session_state['run_time']+1
# st.write("run time "+str(st.session_state['run_time']))
# st.write("Outside Login time "+str(st.session_state['login_time']))
# st.write(st.session_state)
# st.write(st.session_state['login_status'])

# App
# choice = 'Login'

if st.session_state['login_status'] == 'No':
    st.write("enter login now")
    choice_container.title("Please login to start:")
    choice = choice_container.selectbox('login/Signup', ['Login', 'Sign up'],index=0, on_change=clear_state)
    # st.write(choice)
    # Sign up Block
    if (choice == 'Sign up'):
        with signup_container.form("signup_form"):
            email = st.text_input('Please enter your email address')
            password = st.text_input('Please enter your password', type = 'password')    
            username = st.text_input('Please input your user name', value='')
            company = st.text_input('Please input your company name', value='')
            signup = st.form_submit_button('Create account')
        if signup:
            try:
                user = auth.create_user_with_email_and_password(email, password)
                db.child(user['localId']).child("ID").set(user['localId'])
                db.child(user['localId']).child("Username").set(username)
                db.child(user['localId']).child("Company").set(company)
                db.child(user['localId']).child("Email").set(email)
                db.child(user['localId']).child("Password").set(password)

                st.session_state['login_status'] = "Yes"
                st.session_state['user'] = user
                st.session_state['username'] = username
                st.session_state['email'] = email
                # st.session_state['choice_bar'] = 'Login'
                choice_place.empty()
                signup_place.empty()
                # signup_container.success('Your account is created suceesfully!')
                # signup_container.title('Welcome ' + st.session_state['username'])
                # st.balloons()
                # st.experimental_rerun()
            except:
                signup_container.write('Unable to signup user, please try anther email')
                st.experimental_rerun()

    # Login Block
    if (choice == 'Login'):
        with login_container.form("login_form"):
            email = st.text_input('Please enter your email address')
            password = st.text_input('Please enter your password',type = 'password')
            login_form = st.form_submit_button('Login_frontend')
        if login_form:
                
            try:
                user = auth.sign_in_with_email_and_password(email,password)
                # print('login success now 1')
                username = db.child(user['localId']).child("Username").get().val()
                # user_view = auth.get_account_info()
                # st.write(user_view)
                db.child(user['localId']).child("Password").set(password)
                # print('login success now 2')
                
                st.session_state['login_status'] = "Yes"
                st.session_state['user'] = user
                st.session_state['username'] = username
                st.session_state['email'] = email
                # print('login success now 3')
                
                choice_place.empty()
                login_place.empty()
                # login_container.title('Welcome ' + st.session_state['username'])
                # st.balloons()
                # st.session_state['login_time'] = st.session_state['login_time']+1
                # st.write("Inside Login time "+str(st.session_state['login_time']))
                # st.write(st.session_state)
                # st.stop()
                # st.experimental_rerun()
            except:
                # st.write('I am in except status')
                st.write('User not found, please try again. If you are a new user, please create an account.')
                # st.session_state['login_time'] = st.session_state['login_time']+1
                # st.write(st.session_state)
                # st.write(st.session_state)
                # st.stop()
        
# End of Authentication -----------------------------------------------------------------------------------------------------

# Begin of Mainpage after login ---------------------------------------------------------------------------------------------------
if st.session_state['login_status'] == 'Yes':
    # choice_place.empty()
    # st.write(st.session_state)
    user = st.session_state['user']
    username = st.session_state['username']
    email = st.session_state['email']

# Start Navigation menu ---------------------------------------------------------------------------------------------------
    # bio = st.radio('Jump to',['Home','Calculation'])
    # st.sidebar.markdown("""---""")
    menu_holder = st.sidebar.empty()
    menu = menu_holder.container()
    
    with menu:
        select = option_menu(None, ["Home", "Calculation", "Prediction", 'Settings','Log Out','Reset Password'], 
        icons=['house', 'cloud-upload', "list-task", 'gear','gear','gear'], 
        menu_icon="cast", default_index=0, orientation="vertical")  
     
    # st.write("enter menu")
    # st.write(st.session_state)
    
    if select == 'Log Out':
        clear_state()
        st.experimental_rerun()
        
    if select == 'Reset Password':
        auth.send_password_reset_email(email)
        st.success("Successful reset password")
        clear_state()
        st.experimental_rerun()     
    
    if select == 'Calculation':
        st.title('Attrition Analytics')
        st.write('We are building an analytics platform to better understand turnover risk')
        with st.form("my_form"):
            cal_start = st.number_input('Insert a number')
            cal_add = st.number_input('Add a number')
            cal_submit = st.form_submit_button("Submit")
        if cal_submit:
            cal_result = cal_start+cal_add
            st.write('your final number is: '+str(cal_result))
            st.session_state['cal_result'] = cal_result
        
        butt_save = st.button('Save result')
        if butt_save:
            cal_result = st.session_state['cal_result']
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
            cal_save = {'Calculation' : cal_result,
                    'Timestamp' : dt_string} 

            db.child(user['localId']).child("Calculation").push(cal_save)
            st.balloons()

        # st.write(st.session_state)
        
    elif select == 'Home':
        st.write("You are home!")
        exclude_list = ['login_status','user','username','data']
        uploaded_file = st.file_uploader('Step 1: Upload Data Template', type=['xlsx'], on_change=clear_file,args=[exclude_list])
        df = None
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file,sheet_name="Sheet1")
            df['price_update'] = df['price']*100
            st.session_state.data = df
            st.session_state.upload_status = "Yes"
            
            #result, model = analysis(df)
            
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
            
#             save_path = uploaded_file
            
#             # uid = user['localId']
#             fireb_upload = storage.child(user['localId']).put(save_path,user['idToken'])
#             data_url = storage.child(user['localId']).get_url(fireb_upload['downloadTokens']) 
#             db.child(user['localId']).child("Data").child("Store_URL").push(data_url)

#             st.write(save_path)
            st.balloons()
        
        butt_load_data = st.button('Load data')
        # butt_load_file = st.button('Load file')
        
        if butt_load_data:
            data_all = db.child(user['localId']).child("Data").get().val()
            if data_all is not None:
                val = db.child(user['localId']).child("Data").get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                val_by_time = db.child(user['localId']).sort(val, "Timestamp")
                myfile = val_by_time.each()[0].val()
                st.write(len(val_by_time.each()))
                st.write(myfile['Filename'])
                st.write(myfile['Timestamp'])
                st.write(pd.DataFrame.from_dict(myfile['File']))
                download = pd.DataFrame.from_dict(myfile['File']).to_excel('download.xlsx')
                
                # for myfile in val_by_time.each():
                #     myfile = myfile.val()
                #     # if myfile['Filename'] == 'first':
                #     #     download = pd.DataFrame.from_dict(myfile['File']).to_excel('download.xlsx')
                #     st.write(myfile['Filename'])
                #     st.write(myfile['Timestamp'])
                #     st.write(pd.DataFrame.from_dict(myfile['File']))
        
#         if butt_load_file:
#             file_all = db.child(user['localId']).child("Data").child("Store_URL").get().val()
#             if file_all is not None:
#                 val = db.child(user['localId']).child("Data").child("Store_URL").get()
#                 mydata = val.each()[0]
#                 # load_data = mydata.val()+'?raw=true'
#                 load_data = mydata.val()
#                 st.write(load_data)
                
#                 # get_content = requests.get(load_data).content
#                 # read_data = pd.read_excel(io.StringIO(get_content.decode('utf-8')))
#                 # read_data = pd.read_csv(get_content, encoding= 'unicode_escape')
#                 read_data = pd.read_excel(load_data,engine='openpyxl')
#                 # read_data = pd.read_excel(load_data)
#                 st.write(read_data)
                
#                 # get_content = requests.get(load_data).content
#                 # r = requests.get(load_data)
#                 # open('temp.xls', 'wb').write(r.content)
#                 # read_data = pd.read_excel('temp.xls').122294654120....................................8551545545514545666----------------------------------------------------------------------------




                
                # read_data = pd.read_excel(load_data,engine='openpyxl')
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
        
        # st.write(st.session_state)
