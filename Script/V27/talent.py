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
import operator
import copy

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

# Streamlit - format buttons
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
m = st.markdown("""
    <style>
    div.stButton > button:first-child {box-shadow: 0px 0px 0px 2px #3498DB;background-color:#3498DB;border-radius:5px;border:2px solid #3498DB;display:inline-block;cursor:pointer;color:#ffffff;font-family:Arial;font-size:13px;padding:8px 25px;text-decoration:none;
    &:active {position:relative;top:1px;}}
    </style>""", unsafe_allow_html=True)

# Streamlit - hide top menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Streamlit - Set sidebar size
st.markdown(
     """
     <style>
     [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
         width: 270px;
       }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
           width: 270px;
           margin-left: -270px;
       }
    </style>
    """,unsafe_allow_html=True)

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
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    writer.save()
    processed_data = output.getvalue()
    bin_str = base64.b64encode(processed_data).decode()
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

def choose_run_change():
    st.session_state['choose_fullrun_index'] = 1 - st.session_state['choose_fullrun_index']

def feature_1_index_change(feature_1_change,col_list):
    st.session_state['feature_1_index'] = col_list.index(feature_1_change)
    
def cut_1_text_default_change(cut_1):
    st.session_state['cut_1_text_default'] = cut_1

def cut_1_numeric_default_change(cut_1):
    st.session_state['cut_1_range'] = [cut_1.min(),cui_1.max()]
    
# Streamlit initialize session state to track login, upload status, data, and others ------------------------------------------------ 
if 'login_time' not in st.session_state:
    st.session_state['login_time'] = 0
    
if 'run_time' not in st.session_state:
    st.session_state['run_time'] = 0

if 'login_status' not in st.session_state:
    st.session_state['login_status'] = 'No'

if 'data' not in st.session_state:
    st.session_state['data'] = None

if 'data_type' not in st.session_state:
    st.session_state['data_type'] = None

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
    
if 'choose_fullrun' not in st.session_state:
    st.session_state['choose_fullrun'] = None

if 'choose_fullrun_index' not in st.session_state:
    st.session_state['choose_fullrun_index'] = 0
    
if 'feature_1' not in st.session_state:
    st.session_state['feature_1'] = None    
    
if 'feature_1_index' not in st.session_state:
    st.session_state['feature_1_index'] = 0

if 'cut_1_text' not in st.session_state:
    st.session_state['cut_1_text'] = None
    
if 'cut_1_text_default' not in st.session_state:
    st.session_state['cut_1_text_default'] = None
    
if 'cut_1_numeric' not in st.session_state:
    st.session_state['cut_1_numeric'] = None
    
if 'cut_1_range' not in st.session_state:
    st.session_state['cut_1_range'] = None
    
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
    login_container.title("Welcome to TalentX")
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
            login_form = st.form_submit_button('Login')
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
        select = option_menu(username, ["Setup", "Insight", "Prediction", 'Log Out','Reset Password'], 
        icons=['house', 'bar-chart-line', "list-task", 'gear','arrow-clockwise'], 
        menu_icon="person", default_index=0, orientation="vertical",
        styles={
        "container": {"padding": "0!important"},
        "icon": {"font-size": "20px"},
        "nav-link-selected": {"background-color": "#3498DB"}})  

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
        
    # Step 1: Download instruction and template
        step1_col1, step1_col2 = setup_container.columns((1, 5))
        step1_col1.image('Image/step1.jpg',use_column_width='auto')
        # step1_col1.image('Image/step1.jpg',width=200)
        step1_col2.markdown("🖱️ 'Save link as...'")
        step1_col2.markdown(get_binary_file_downloader_html(file_path, 'Instruction and Template'), unsafe_allow_html=True)
        setup_container.markdown("""---""")
        
    # Step 2: Submit data
        # exclude_list = ['login_status','user','username','email','menu_message','data']
        # step2_col1, step2_col2 = setup_container.columns((1, 5))
        step2_col1, step2_col2 = setup_container.columns((1, 5))
        exclude_list = ['login_status','user','username','email','menu_message']
        step2_col1.image('Image/step2.jpg',use_column_width='auto')
        uploaded_file = step2_col2.file_uploader('', type=['xlsx'], on_change=clear_state_withexc,args=[exclude_list])
        setup_container.write(st.session_state['choose_fullrun_index'])
        df = None
        if (uploaded_file is not None) and (st.session_state['upload_status'] == "No"):
            df = pd.read_excel(uploaded_file,sheet_name="Submission")
            df_type = df.iloc[0].tolist()
            # df_type.columns=['col_type']
            df = df.iloc[1:,:]
            # df_type.to_excel('type.xlsx')
            df.to_excel('df.xlsx')
            
            # df['price_update'] = df['price']*100
            print(df_type)
            st.session_state['data'] = df
            st.session_state['data_type'] = df_type
            st.session_state['upload_status'] = "Yes"
            
        elif st.session_state['upload_status'] == "Yes":
            df = st.session_state['data']
            df_type = st.session_state['data_type']
        setup_container.markdown("""---""")
        
        if df is not None:
            step3_col1, step3_col2 = setup_container.columns((1, 5))
            # Yang - function validation(df)
            # Call a function to pass df and return with a output dictionary
            output = {'validation':{'Submitted Entry':1470, 'Processed Entry':1400, 'Imputed Entry':10 ,'Invalid Entry':60, 
                      'Invalid Data': df.tail(60), 'Processed Data': df.head(1410),
                      'All Valid Columns':df.columns.tolist(), 'All Valid Columns and Types':dict(zip(df.columns.tolist(), df_type))        
                     }}
            df_clean = output['validation']['Processed Data']
            # df_col_list = output['validation']['All Valid Columns'] 
            df_col_dict = output['validation']['All Valid Columns and Types']
            df_col_list = list(df_col_dict.keys())
            
            # End of Yang
            
    # Step 3: Data Validation
            # step3_col2.write('Step 3: Validate Data')
            step3_col1, validation_col1,validation_col2,validation_col3,validation_col4, validation_col5 = setup_container.columns((1, 1, 1, 1, 1, 1))
            step3_col1.image('Image/step3.jpg',use_column_width='auto')
            validation_col1.metric('Submitted Entry',output['validation']['Submitted Entry'])
            validation_col2.metric('Processed Entry',output['validation']['Processed Entry'])
            validation_col3.metric('Imputed Entry',output['validation']['Imputed Entry'])
            validation_col4.metric('Invalid Entry',output['validation']['Invalid Entry'])
            df_validation = output['validation']['Invalid Data']
            if operator.not_(df_validation.empty):
                validation_col5.markdown(get_excel_file_downloader_html(df_validation, 'Invalid Entry.xlsx'), unsafe_allow_html=True)
                validation_col5.markdown("🖱️ 'Save link as...'")
            setup_container.markdown("""---""")
            
    # Step 4: Data Validation
            step4_col1, step4_col2, step4_col3, step4_col4, step4_col5 = setup_container.columns((1, 2, 1, 1, 1))
            step4_col1.image('Image/step3.jpg',use_column_width='auto')
            choose_run = step4_col2.radio('Would you like to analyse entire population?',('Yes', 'No'), index = st.session_state['choose_fullrun_index'], on_change = choose_run_change)
            
            st.session_state['choose_fullrun'] = choose_run
            if st.session_state['choose_fullrun'] == 'No':
                st.session_state['choose_fullrun_index'] = 1
            else:
                st.session_state['choose_fullrun_index'] = 0

    # Step 4a: Start - Run a segment of population - Allow user to choose up to 3 filters (relationship is A and B and C)
    # If user select yes, use the entire dataset
            df_final = copy.deepcopy(df_clean)
            if st.session_state['choose_fullrun'] == 'No':
                step4a_col1, step4a_col2, step4a_col3 = setup_container.columns((1, 1, 1))    
                
            #1st cut
                feature_1 = step4a_col1.selectbox('1st Cut',output['validation']['All Valid Columns'], index = st.session_state['feature_1_index'])
                # Update default selection
                feature_1_index_change(feature_1,df_col_list)
                st.session_state['feature_1'] = feature_1
                # setup_container.write(st.session_state['feature_1'])
                # setup_container.write(feature_1)
                
                if df_col_dict[feature_1] == 'text':
                    cut_1 = step4a_col1.multiselect(feature_1,set(df_final[feature_1]), key='choose_cut1', 
                                                    default=st.session_state['cut_1_text_default'])
                    cut_1_text_default_change(cut_1)
                    st.session_state['cut_1_text'] = cut_1
                    df_final = df_final[df_final[feature_1].isin(cut_1)]
                elif df_col_dict[feature_1] == 'numeric':
                    range_min = df_final[feature_1].min()
                    range_max = df_final[feature_1].max()
                    cut_1 = step4a_col1.slider('Select a range of values',range_min,range_max, 
                                               st.session_state['cut_1_range'])
                    cut_1_numeric_default_change(cut_1)
                    st.session_state['cut_1_numeric'] = cut_1
                    df_final = df_final[(df_final[feature_1]>=min(cut_1)) & (df_final[feature_1]<=max(cut_1))]
            #2nd cut
                if len(cut_1)>0:
                    # cut_1_message = feature_1+' includes '+', '.join(cut_1)
                    # setup_container.info('Run a subset where '+cut_1_message)
                    # df_temp_1.to_excel('temp1.xlsx')
                    feature_2 = step4a_col2.selectbox('2nd Cut',output['validation']['All Valid Columns'])
                    # setup_container.write(set(df_final[feature_2]))
                    cut_2 = step4a_col2.multiselect(feature_2,set(df_final[feature_2]),key='choose_cut2')
                    df_final = df_final[df_final[feature_2].isin(cut_2)]
            #3rd cut    
                    if len(cut_2)>0:
                        # cut_2_message = feature_2+' includes '+', '.join(cut_2)
                        # setup_container.info('Run a subset where '+cut_1_message+' and '+cut_2_message)
                        
                        df_temp_2 = df_clean[df_clean[feature_2].isin(cut_2)]
                        # df_temp_1.to_excel('temp1.xlsx')
                        feature_3 = step4a_col3.selectbox('3rd Cut',output['validation']['All Valid Columns'])
                        # setup_container.write(set(df_temp_2[feature_3]))
                        cut_3 = step4a_col3.multiselect(feature_3,set(df_temp_2[feature_3]),key='choose_cut3')
                        df_final = df_final[df_final[feature_3].isin(cut_3)]
                        
                        if len(cut_3)>0:
                            print('a')
                            # cut_3_message = feature_3+' includes '+', '.join(cut_3)
                            # setup_container.info('Run a subset where '+cut_1_message+' and '+cut_2_message +' and '+cut_3_message)
    # Step 4a. End - output a user selected defined dataset
            setup_container.markdown("""---""")
            # if len(cut_1)>0:
            df_final.to_excel('final_data.xlsx')
    
    
            
                
                
                # with step4_col3.form("user selection"):
                #     feature_1 = st.selectbox('1st Cut',output['validation']['All Valid Columns'])
                #     cut_1 = st.multiselect(feature_1,set(df[feature_1]))
                #     cut_submit = st.form_submit_button("Submit")
                
                # cut2 = st.selectbox('1st Cut',('Email', 'Home phone', 'Mobile phone'))
                # cut3 = st.selectbox('1st Cut',('Email', 'Home phone', 'Mobile phone'))
                
                
           # Step 5: Run a segment of population - Allow user to choose up to 3 filters
                # step5_col1, step5_col2, step5_col3, step5_col4 = setup_container.columns((1, 2, 2, 2))
                # step5_col1.image('Image/step3.jpg',use_column_width='auto')
                
        
        
#         df_name = setup_container.text_input('Filename', '')
        
#         butt_save_data = setup_container.button('Save data')
        
#         if butt_save_data:
#             # setup_container.write(df.to_dict())
#             st.write(df.head(1))
#             now = datetime.now()
#             dt_string = now.strftime("%d/%m/%Y %H:%M:%S")              
#             data_save = {'Filename':df_name,
#                     'File' : df.to_dict(),
#                     'Timestamp' : dt_string} 
#             db.child(user['localId']).child("Data").push(data_save)
            
#             # output = BytesIO()
#             # writer = pd.ExcelWriter(output, engine='xlsxwriter')
#             # df.to_excel(writer, index=False, sheet_name='Sheet1')
#             # workbook = writer.book
#             # worksheet = writer.sheets['Sheet1']
#             # writer.save()
#             # processed_data = output.getvalue()
            
#             # save_path = get_excel_file_downloader_url(processed_data, 'save_data.xlsx')
#             # save_path = 'Test/test.xlsx'
            
# #             save_path = uploaded_file
            
# #             # uid = user['localId']
# #             fireb_upload = storage.child(user['localId']).put(save_path,user['idToken'])
# #             data_url = storage.child(user['localId']).get_url(fireb_upload['downloadTokens']) 
# #             db.child(user['localId']).child("Data").child("Store_URL").push(data_url)

# #             st.write(save_path)
#             st.balloons()
        
#         butt_load_data = setup_container.button('Load data')
#         # butt_load_file = st.button('Load file')
        
#         if butt_load_data:
#             data_all = db.child(user['localId']).child("Data").get().val()
#             if data_all is not None:
#                 val = db.child(user['localId']).child("Data").get()
#                 # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
#                 # val = db.child(user['localId']).child("Data").order_by_child('Timestamp').get()
#                 val_by_time = db.child(user['localId']).sort(val, "Timestamp")
#                 myfile = val_by_time.each()[0].val()
#                 setup_container.write(len(val_by_time.each()))
#                 setup_container.write(myfile['Filename'])
#                 setup_container.write(myfile['Timestamp'])
#                 setup_container.write(pd.DataFrame.from_dict(myfile['File']))
#                 download = pd.DataFrame.from_dict(myfile['File']).to_excel('download.xlsx')
                
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
        
        insight_place = st.empty()
        insight_container = insight_place.container()
        
        
        with insight_container.form("my_form"):
            cal_start = st.number_input('Insert a number')
            cal_add = st.number_input('Add a number')
            cal_submit = st.form_submit_button("Submit")
        if cal_submit:
            cal_result = cal_start+cal_add
            st.write('your final number is: '+str(cal_result))
            st.session_state['cal_result'] = cal_result
        
        butt_save = insight_container.button('Save result')
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
