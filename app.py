import pandas as pd
import streamlit as st
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import joblib
from joblib import dump, load

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
column_names2 =['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']
number_list =[]
l=np.ones(shape=(1,9))
number_df=pd.DataFrame(l,columns=column_names)
# all_answers =[]

st.write("Q1: I worry a lot more since the stressful event(s)")
number = st.number_input('Q1', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)

st.write("Q2:  I cannot stop thinking about the stressful event(s)")
number = st.number_input('Q2', min_value=0, max_value=4, step=1,  key=f'qq_2')
number_list.append(number)

st.write("Q3: I often feel afraid about what might happen in the future since the stressful event(s) (preoccupation).")
number = st.number_input('Q3', min_value=0, max_value=4, step=1,  key=f'qq_3')
number_list.append(number)

st.write("Q4:I find it difficult to adapt to life since the stressful event(s) ")
number = st.number_input('Q4', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)
st.write("Q5: I find it difficult to relax and feel calm since the stressful event(s)")
number = st.number_input('Q5', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)
st.write("Q6: I find it difficult to achieve a state of inner peace since the stressful event(s) ")
number = st.number_input('Q6', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)
st.write("Q7: Affected your relationships or social life?")
number = st.number_input('Q7', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)
st.write("Q8:  Affected your ability to work or your educational life? ")
number = st.number_input('Q8', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)
st.write("Q9: Affected any other important part of your life?")
number = st.number_input('Q9', min_value=0, max_value=4, step=1,  key=f'qq_1')
number_list.append(number)



# col1, col2, col3,col4, col5, col6, col7, col8, col9 = st.columns(9)
# with col1:
#   st.header("Pr1")
#   st.write("Q1: I worry a lot more since the stressful event(s)")
#   number = st.number_input('Q1', min_value=0, max_value=4, step=1,  key=f'q_1')
#   number_list.append(number)

# with col2:
#   st.header("Pr2")
#   number = st.number_input("Pr2",min_value=0, max_value=4, step=1, key=f'q_2')
#   number_list.append(number)
# with col3:

#   st.header("Pr3")
#   number = st.number_input("Pr3",min_value=0, max_value=4, step=1, key=f'q_3')
#   number_list.append(number)
# with col4:

#   st.header("FA1")
#   number = st.number_input('FA1',min_value=0, max_value=4, step=1, key=f'q_4')
#   number_list.append(number)
# with col5:

#   st.header("FA2")
#   number = st.number_input("FA2",min_value=0, max_value=4, step=1, key=f'q_5')
#   number_list.append(number)
# with col6:

#   st.header("FA3")
#   number = st.number_input("FA3",min_value=0, max_value=4, step=1, key=f'q_6')
#   number_list.append(number)
# with col7:

#   st.header("FI1")
#   number = st.number_input('FI1',min_value=0, max_value=4, step=1, key=f'q_7')
#   number_list.append(number)
# with col8:

#   st.header("FI2")
#   number = st.number_input("FI2",min_value=0, max_value=4, step=1, key=f'q_8')
#   number_list.append(number)
# with col9:

#   st.header("FI3")
#   number = st.number_input("FI3",min_value=0, max_value=4, step=1, key=f'q_9')
#   number_list.append(number)

l=np.ones(shape=(1,9))
number_df=pd.DataFrame(l,columns=column_names2)

for i in range(l.shape[1]):
  number_df.iloc[:,i]= number_list[i]
st.write(number_df)
number_df.rename(columns={'Q1':'Pr1','Q2':'Pr2','Q3':'Pr3','Q4': 'FA1','Q5':'FA2','Q6':'FA3','Q7': 'FI1',
                  'Q8':'FI2','Q9':'FI3'}, inplace=True)

clf_loaded = load('knn.joblib')

classifier=st.button("Am I Honest or Dishonest?")
if classifier:  
  y_pred=clf_loaded.predict(number_df)
  if y_pred[0]=='H':
    st.write("You are Honest!")
  else:
    st.write("You are Dishonest")

