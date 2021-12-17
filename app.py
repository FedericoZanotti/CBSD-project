import pandas as pd
import streamlit as st
import numpy as np
import joblib
from joblib import dump, load
import sklearn
from sklearn.neighbors import KNeighborsClassifier
st.title("Welcome to noncagarmeercazzo!")

st.text("In questa demo potrai caricare un tuo file CSV con le risposte del questionario IADQ o scrivere direttamente le risposte nella sezione dopo")

# st.title("Button")
# result = st.button("Click here")
# st.write("result",result)
# if result:
#   st.write(":smile:")
st.header("Load your file")
dataframe_loaded=pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     dataframe_loaded = pd.read_csv(uploaded_file)
     st.write(dataframe_loaded)

st.header("Load your answers")
# answers=st.text_input("How many people answers do you want to write?")
# st.write(f"answers {answers}")
column_names=['Pr1','Pr2','Pr3', 'FA1','FA2','FA3', 'FI1','FI2','FI3']
number_list =[]
l=np.ones(shape=(1,9))
number_df=pd.DataFrame(l,columns=column_names)
# all_answers =[]
col1, col2, col3,col4, col5, col6, col7, col8, col9 = st.columns(9)
with col1:
  st.header("Pr1")
  number = st.number_input('Pr1', min_value=0, max_value=4, step=1,  key=f'q_1')
  number_list.append(number)
with col2:
  st.header("Pr2")
  number = st.number_input("Pr2",min_value=0, max_value=4, step=1, key=f'q_2')
  number_list.append(number)
with col3:

  st.header("Pr3")
  number = st.number_input("Pr3",min_value=0, max_value=4, step=1, key=f'q_3')
  number_list.append(number)
with col4:

  st.header("FA1")
  number = st.number_input('FA1',min_value=0, max_value=4, step=1, key=f'q_4')
  number_list.append(number)
with col5:

  st.header("FA2")
  number = st.number_input("FA2",min_value=0, max_value=4, step=1, key=f'q_5')
  number_list.append(number)
with col6:

  st.header("FA3")
  number = st.number_input("FA3",min_value=0, max_value=4, step=1, key=f'q_6')
  number_list.append(number)
with col7:

  st.header("FI1")
  number = st.number_input('FI1',min_value=0, max_value=4, step=1, key=f'q_7')
  number_list.append(number)
with col8:

  st.header("FI2")
  number = st.number_input("FI2",min_value=0, max_value=4, step=1, key=f'q_8')
  number_list.append(number)
with col9:

  st.header("FI3")
  number = st.number_input("FI3",min_value=0, max_value=4, step=1, key=f'q_9')
  number_list.append(number)

l=np.ones(shape=(1,9))
number_df=pd.DataFrame(l,columns=column_names)

for i in range(l.shape[1]):
  number_df.iloc[:,i]= number_list[i]

st.write(number_df)

clf_loaded = load('knn.joblib')

classifier=st.button("Am I Honest or Dishonest?")
if classifier:  
  y_pred=clf_loaded.predict(number_df)
  if y_pred[0]=='H':
    st.write("You are Honest!")
  else:
    st.write("You are Dishonest")

