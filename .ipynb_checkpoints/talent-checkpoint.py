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

# Setup Filepath for user submission data template
file_path = Path(__file__).parents[0].__str__()+'/Data/Data Template.xlsx'

# Firebase Authentication----------------------------------------------------------------------------------------------------------------------
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
# Download Excel Template
@st.experimental_memo(show_spinner=False)
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    # href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">{file_label}</a>'
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Download Excel Files
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

def clear_state_withexc(exclude_list):
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
    
if 'menu_message' not in st.session_state:
    st.session_state['menu_message'] = None
    
# Streamlit -----------------------------------------------------------------------------------------------------------------------

# Authentication -----------------------------------------------------------------------------------------------------------------------
# choice_place = st.empty()
# choice_container = choice_place.container()

# Obtain User Input for email and password
login_place = st.empty()
login_container = login_place.container()

# App
# choice = 'Login'
if st.session_state['login_status'] == 'No':
    # st.write("enter login now")
    if st.session_state['menu_message'] is not None:
        login_container.info(st.session_state['menu_message'])
    login_container.title("Welcome to Talent Analytics")
    login_col1, login_col2, login_col3 = login_container.columns([1,1,0.2])
    login_col1.image('Image/login3.jpg',use_column_width='auto')
    choice = login_col2.selectbox('login/Signup', ['Login', 'Sign up'],index=0, on_change=clear_state) 
    # st.write(choice)
    # Sign up Block
    if (choice == 'Sign up'):
        with login_col2.form("signup_form"):
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
                st.experimental_rerun()
            except:
                st.write('Unable to signup user, please try anther email')
                st.experimental_rerun()

    # Login Block
    if (choice == 'Login'):
        with login_col2.form("login_form"):
            email = st.text_input('Please enter your email address')
            password = st.text_input('Please enter your password',type = 'password')
            login_form = st.form_submit_button('Login Account')
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
                login_place.empty()

            except:
                st.write('User not found, please try again. If you are a new user, please create an account.')
        
# End of Authentication -----------------------------------------------------------------------------------------------------

# Begin of Mainpage after login ---------------------------------------------------------------------------------------------------
if st.session_state['login_status'] == 'Yes':
    user = st.session_state['user']
    username = st.session_state['username']
    email = st.session_state['email']

# Start Navigation menu ---------------------------------------------------------------------------------------------------
    menu_holder = st.sidebar.empty()
    menu = menu_holder.container()
    
    with menu:
        select = option_menu("Welcome "+username, ["Setup", "Insight", "Prediction", 'Log Out','Reset Password'], 
        icons=['house', 'bar-chart-line', "list-task", 'gear','arrow-clockwise'], 
        menu_icon="person", default_index=0, orientation="vertical")  
        
    if select == 'Setup':
        setup_place = st.empty()
        setup_container = setup_place.container()
        
        setup_col1, setup_col2 = setup_container.columns((2, 1))
        setup_col1.title('TalentX')
        setup_col1.write('TalentX is an AI-driven platform to analyzing employee turnover to identify why people leave and boost retention. By analying termination data during 12 month period, TalentX can identify root cause of turnover, answer what if questions, and predict individual turnover risk for next 12 month.')
        setup_col2.image('Image/setup1.jpg',use_column_width='auto')
        
        # Start setup
        setup_container.markdown("""---""")
        # setup_container.markdown("🎯 Let's Get Started")
        
        # Step 1
        step1_col1, step1_col2, step1_col3 = setup_container.columns((1, 1, 3))
        step1_col1.markdown("Step 1: 🖱️ 'Save link as...'")
        step1_col2.markdown(get_binary_file_downloader_html(file_path, 'Instruction and Template'), unsafe_allow_html=True)
        
        exclude_list = ['login_status','user','username','data']
        uploaded_file = setup_container.file_uploader('Step 2: Upload Data Template', type=['xlsx'], on_change=clear_state_withexc,args=[exclude_list])
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
            setup_container.write(df['price'][0])
            setup_container.write(df['price_update'][0])
        
        df_name = setup_container.text_input('Filename', '')
        
#         butt_next_step = setup_container.button('My Insight')
#         if butt_next_step:
            
            
        
        butt_save_data = setup_container.button('Save data')
        
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
        
        butt_load_data = setup_container.button('Load data')
        # butt_load_file = st.button('Load file')
        
        if butt_load_data:
            data_all = db.child(user['localId']).child("Data").get().val()
            if data_all is not None:
                val = db.child(user['localId']).child("Data").get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
                val_by_time = db.child(user['localId']).sort(val, "Timestamp")
                myfile = val_by_time.each()[0].val()
                setup_container.write(len(val_by_time.each()))
                setup_container.write(myfile['Filename'])
                setup_container.write(myfile['Timestamp'])
                setup_container.write(pd.DataFrame.from_dict(myfile['File']))
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
    elif select == 'Insight':
        # st.title('Attrition Analytics')
        # st.write('We are building an analytics platform to better understand turnover risk')
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

    elif select == 'Log Out':
        clear_state()
        st.experimental_rerun()
        
    elif select == 'Reset Password':
        auth.send_password_reset_email(email)
        st.session_state['menu_message'] = "A password reset message was sent. Click the link in the email to create a new password."
        clear_state_withexc('menu_message')
        st.experimental_rerun()  
                
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
