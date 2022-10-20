import streamlit as st
import pandas as pd
import requests

def run():

    st.title("Fraud Detection")
    st.subheader("For single sample, kindly fill the below form")
    with st.form("my_form"):

        V1 = st.text_input(label="V1")
        V2 = st.text_input(label="V2")
        V3 = st.text_input(label="V3")
        V4 = st.text_input(label="V4")
        V5 = st.text_input(label="V5")
        V6 = st.text_input(label="V6")
        V7 = st.text_input(label="V7")
        V8 = st.text_input(label="V8")
        V9 = st.text_input(label="V9")
        Amount = st.text_input(label="Amount")
        form_lst = [V1, V2, V3, V4, V5, V6, V7, V8, V9, Amount]
        df = pd.DataFrame(form_lst)
        submitted = st.form_submit_button("Submit")
    if submitted:
        st.write(df)
        st.write("Your sample is submitted")



    st.subheader("For multiple samples, kindly upload the csv file")
    file_up = st.file_uploader("Upload below", type = ['csv'])
    if file_up is not None:
        print("hello")
        dg = pd.read_csv(file_up)
        independent_vars = dg
        dependent_var = dg['Class']
        independent_vars.drop('Class', inplace=True, axis=1)

        independent_vars.drop(['Time', 'V10','V11','V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                               'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], inplace = True, axis=1)

        st.dataframe(independent_vars.head(5))
        st.write("Upload success!")
        

    data = {'features': form_lst}
    if st.button('predict'):
        response = requests.post("http://0.0.0.0:8000/predict", json="")

        st.success(response.text)


if __name__ == "__main__":
    run()
