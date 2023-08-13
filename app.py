import streamlit as st
from streamlit_option_menu import option_menu
import eda
import prediction
import about

st.write('### PREDICTION OF CREDIT CARD DEFAULT')
st.write('##### This page created by [Theo Nugraha](https://github.com/theonugraha)')
st.markdown('---')

selected = option_menu(None, ["About", "EDA", "Predict"], 
    icons=['house', 'bar-chart', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"1px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "grey"},
    }
)
    
selected
    

if selected == 'EDA':
    eda.run()
elif selected == 'Predict':
    prediction.run()
else:
    about.run()