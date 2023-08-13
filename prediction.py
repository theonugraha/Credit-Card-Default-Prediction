import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load all files
with open('list_cat_cols.txt', 'r') as file_1:
  list_cat_cols = json.load(file_1)

with open('list_num_cols.txt', 'r') as file_2:
  list_num_cols = json.load(file_2)

with open('model_scaler.pkl', 'rb') as file_3:
  scaler = pickle.load(file_3)

with open('model_encoder.pkl', 'rb') as file_4:
  encoder = pickle.load(file_4)

with open('model_svc.pkl', 'rb') as file_5:
  model_svc = pickle.load(file_5)
  
with open('model_grid_rf.pkl', 'rb') as file_6:
  model_grid_rf = pickle.load(file_6)
  
def run():
    st.write('##### Form Prediction Credit Card Default')
      
    # Making Form
    with st.form(key='Form Prediction Credit Card Default'):
        Limit_Balance       = st.number_input('limit_balance', min_value=10000, max_value=800000, value=10000, step=1, help='Limit Balance')
        Sex                 = st.selectbox('sex', (1,2), index=1, help='1=Male, 2=Female')
        Education_Level     = st.selectbox('education_level', (1,2,3,4), index=1, help='1=Graduate School, 2=University, 3=High School, 4=Others')
        Marital_Status      = st.selectbox('marital_status', (1,2,3), index=1, help='1=married, 2=single, 3=others')
        Age                 = st.number_input('age', min_value=20, max_value=70, value=22)
        st.markdown('---')
        Pay_0               = st.number_input('pay_0', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        Pay_2               = st.number_input('pay_2', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        Pay_3               = st.number_input('pay_3', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        Pay_4               = st.number_input('pay_4', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        Pay_5               = st.number_input('pay_5', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        Pay_6               = st.number_input('pay_6', min_value=-1, max_value=12, value=0, help='-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above')
        st.markdown('---')
        Bill_Amount_1       = st.number_input('bill_amt_1', min_value=-80000, max_value=650000, value=0)
        Bill_Amount_2       = st.number_input('bill_amt_2', min_value=-80000, max_value=650000, value=0)
        Bill_Amount_3       = st.number_input('bill_amt_3', min_value=-80000, max_value=650000, value=0)
        Bill_Amount_4       = st.number_input('bill_amt_4', min_value=-80000, max_value=650000, value=0)
        Bill_Amount_5       = st.number_input('bill_amt_5', min_value=-80000, max_value=650000, value=0)
        Bill_Amount_6       = st.number_input('bill_amt_6', min_value=-80000, max_value=650000, value=0)
        st.markdown('---')
        Pay_Amount_1        = st.number_input('pay_amt_1', min_value=-0, max_value=650000, value=0)
        Pay_Amount_2        = st.number_input('pay_amt_2', min_value=-0, max_value=650000, value=0)
        Pay_Amount_3        = st.number_input('pay_amt_3', min_value=-0, max_value=650000, value=0)
        Pay_Amount_4        = st.number_input('pay_amt_4', min_value=-0, max_value=650000, value=0)
        Pay_Amount_5        = st.number_input('pay_amt_5', min_value=-0, max_value=650000, value=0)
        Pay_Amount_6        = st.number_input('pay_amt_6', min_value=-0, max_value=650000, value=0)
        
        submited_1 = st.form_submit_button('Predict using SVC')
        submited_2 = st.form_submit_button('Predict using RFC')
        
        data_inf = {
            'limit_balance'      : Limit_Balance,
            'sex'                : Sex,
            'education_level'    : Education_Level,
            'marital_status'     : Marital_Status,
            'age'                : Age,
            'pay_0'              : Pay_0,
            'pay_2'              : Pay_2,
            'pay_3'              : Pay_3,
            'pay_4'              : Pay_4,
            'pay_5'              : Pay_5,
            'pay_6'              : Pay_6,
            'bill_amt_1'         : Bill_Amount_1,
            'bill_amt_2'         : Bill_Amount_2,
            'bill_amt_3'         : Bill_Amount_3,
            'bill_amt_4'         : Bill_Amount_4,
            'bill_amt_5'         : Bill_Amount_5,
            'bill_amt_6'         : Bill_Amount_6,
            'pay_amt_1'          : Pay_Amount_1,
            'pay_amt_2'          : Pay_Amount_2,
            'pay_amt_3'          : Pay_Amount_3,
            'pay_amt_4'          : Pay_Amount_4,
            'pay_amt_5'          : Pay_Amount_5,
            'pay_amt_6'          : Pay_Amount_6
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submited_1:
        #Split between numerical columns and categorical columns
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]
        #Feature scaling and feature encoding
        data_inf_num_scaled = scaler.transform(data_inf_num)
        data_inf_cat_encoded = encoder.transform(data_inf_cat)
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis = 1)
        #Predict using SVC
        y_pred_inf = model_svc.predict(data_inf_final)
        st.write('# Result : ', str(int(y_pred_inf)))
    else:
        #Split between numerical columns and categorical columns
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]
        #Feature scaling and feature encoding
        data_inf_num_scaled = scaler.transform(data_inf_num)
        data_inf_cat_encoded = encoder.transform(data_inf_cat)
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis = 1)
        #Predict using RFC
        y_pred_inf = model_grid_rf.predict(data_inf_final)
        st.write('# Result : ', str(int(y_pred_inf)))
        
if __name__ == '__main__':
    run()