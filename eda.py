import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title = 'Prediction of Credit Card Default',
    layout ='centered',
    initial_sidebar_state='expanded'
)

def run():
    # Sub Header
    st.subheader('EDA for Analizing Dataset Credit Card Default', )

    # Separated Line
    st.markdown('---')

    # Show Data Frame
    st.write('#### Dataset Credit Card Default')
    data = pd.read_csv('h8dsft_P1M1_theo.csv')
    st.dataframe(data)

    # Histogram based user input
    st.write('#### Histogram')
    option = st.selectbox('Choose Column : ', ('limit_balance','age','bill_amt_1', 'bill_amt_2', 'bill_amt_3', 
                                                'bill_amt_4', 'bill_amt_5', 'bill_amt_6', 'pay_amt_1', 'pay_amt_2', 
                                                'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6'))
    fig = plt.figure(figsize=(20, 10))
    sns.histplot(data[option], bins=30, kde=True)
    st.pyplot(fig)
    
    # Pie chart `default_payment_next_month`
    count_data = data["default_payment_next_month"].value_counts()
    # Pie chart
    st.write('#### Pie Chart Default Payment Next Month')
    fig = plt.figure(figsize=(6, 6))
    plt.pie(count_data, labels=["Not Default", "Default"], autopct="%1.1f%%", startangle=90, colors=["lightblue", "lightcoral"])
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can understand that the amount of data in class 0 (Not Default) is more than class 1 (Default).
            ''')

    # Bar Plot 1
    st.write('#### Plot Default Payment Next Month Based on Sex')
    fig = plt.figure(figsize=(15, 5))
    default = sns.countplot(data=data, x="sex", hue="default_payment_next_month")
    for container in default.containers:
        default.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can know that the number of defaults with male gender is more than the number of defaults with female gender. 
            The number of non-defaulters with male gender is more than the number of non-defaulters with female gender.
            ''')
    
    # Bar Plot 2
    st.write('#### Plot Age Based on Sex')
    fig = plt.figure(figsize=(30,25))
    age = sns.countplot(data=data, x="age", hue="sex")
    for container in age.containers:
        age.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can find out that the most credit card customers are female and aged 30 years, totaling 111 customers. 
            The most male customers are also 30 years old and totaling 60 customers.
            ''')

    # Bar Plot 3
    st.write('#### Plot Education Level Based on Sex')
    fig = plt.figure(figsize=(15, 5)) 
    edu = sns.countplot(data=data, x="education_level", hue="sex")

    for container in edu.containers:
        edu.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can understand that the most customers are customers who are still students and are female, as well as customers who are male.
            ''')
    
    # Bar Plot 4
    st.write('#### Plot Marital Status Based on Sex')
    fig = plt.figure(figsize=(15, 5)) 
    mar = sns.countplot(data=data, x="marital_status", hue="sex")

    for container in mar.containers:
        mar.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can understand that customers with female gender and single status are the most customers, as well as customers with male gender.
            ''')
    
    # Bar Plot 5
    st.write('#### Plot Marital Status Based on Default Payment Next Month')
    fig = plt.figure(figsize=(15, 5)) 
    mar_def = sns.countplot(data=data, x="marital_status", hue="default_payment_next_month")

    for container in mar_def.containers:
        mar_def.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can find out that customers who are single are the customers who default the most and the most do not default.
            ''')
    
    # Bar Plot 6
    st.write('#### Plot Age Based on Default Payment Next Month')
    fig = plt.figure(figsize=(30, 25)) 
    mar_age = sns.countplot(data=data, x="age", hue="default_payment_next_month")

    for container in mar_age.containers:
        mar_age.bar_label(container)
    st.pyplot(fig)
    with st.expander("See explanation"):
        st.write('''
            Based on the diagram above, we can find out that customers aged 30 years are the most customers who do not default, while customers aged 27 are the most customers who default.
            ''')
    

if __name__ == '__main__':
    run()