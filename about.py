import streamlit as st
from PIL import Image

def run():
    # Add Picture
    image = Image.open('cc.jpeg')
    st.image(image, caption='Credit Card Default')
    # Title
    st.title('ABOUT THIS PROJECT')
    st.markdown('---')
    st.write('###### This project aims to predict credit card customer defaults. In this project, I used the Support Vector Classifier and Random Forest Classification method to predict credit card defaults with an accuracy rate of approximately 83%. I performed the scaling method using MinMaxScaller and the encoding method with OrdinalEncoder. I also performed the cross validating method and searched for hyperparameters using GridSearchCV.')
    st.markdown('---')
    
    st.write('Feel free to contact me on:')
    st.write('[GITHUB](https://github.com/theonugraha)')
    st.write('or')
    st.write('[LINKEDIN](https://www.linkedin.com/in/nugrahatheo/)')


if __name__ == '__main__':
    run()