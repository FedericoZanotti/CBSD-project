import pandas as pd
import streamlit as st
import numpy as np
import sklearn
import sklearn.neighbors
from sklearn.neighbors import _dist_metrics
from joblib import dump, load
import base64
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
st.title("Welcome to the Demo!")

st.write(" In this demo you can load your file CSV well with the questionnary IADQ's answers or directly write them on this web app")

# st.title("Button")
# result = st.button("Click here")
# st.write("result",result)
# if result:
#   st.write(":smile:")
def csv_downloader(data, title):
	csvfile = data.to_csv(index=False)
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = f"{title}.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

def recon(ar=0,column_names2=[],column_names=[], csv=False):
  clf1 = load('pr1.joblib')
  clf2 = load('pr2.joblib')  
  clf3 = load('pr3.joblib')
  clf4 = load('fa1.joblib')
  clf5 = load('fa2.joblib')
  clf6 = load('fa3.joblib')  
  clf7 = load('fi1.joblib')
  clf8 = load('fi2.joblib')
  clf9 = load('fi3.joblib')
  idx1=[0,2,4,7]
  idx2=[1,2,4,6]
  idx3=[0,1,2,3,]
  idx4=[6,7,8]
  idx5=[0,4]
  idx6=[0,4]
  idx7=[1,3,7,8]
  idx8=[1,3,7,8]
  idx9=[1,2,7,8]
  if not csv:
    ar = np.asarray(ar)
    ar=ar.reshape(1,-1)
    df_reconstructed = pd.DataFrame(data=[[1,2,3,4,5,6,7,8,9]],columns=column_names2)
    res=int(clf1.predict(ar[:,idx1]))
    df_reconstructed['Q1']=res
    df_reconstructed['Q2']=int(clf2.predict(ar[:,idx2]))
    df_reconstructed['Q3']=int(clf3.predict(ar[:,idx3]))
    df_reconstructed['Q4']=int(clf4.predict(ar[:,idx4]))
    df_reconstructed['Q5']=int(clf5.predict(ar[:,idx5]))
    df_reconstructed['Q6']=int(clf6.predict(ar[:,idx6]))
    df_reconstructed['Q7']=int(clf7.predict(ar[:,idx7]))
    df_reconstructed['Q8']=int(clf8.predict(ar[:,idx8]))
    df_reconstructed['Q9']=int(clf9.predict(ar[:,idx9]))
  else:
    data=[]
    pred1 = clf1.predict(ar[:,idx1])
    pred2 = clf2.predict(ar[:,idx2])
    pred3 = clf3.predict(ar[:,idx3])
    pred4 = clf4.predict(ar[:,idx4])
    pred5 = clf5.predict(ar[:,idx5])
    pred6 = clf6.predict(ar[:,idx6])
    pred7 = clf7.predict(ar[:,idx7])
    pred8 = clf8.predict(ar[:,idx8])
    pred9 = clf9.predict(ar[:,idx9])
    for i in range(pred1.shape[0]):
      data.append([pred1[i], pred2[i],pred3[i], pred4[i],pred5[i], pred6[i],pred7[i], pred8[i],pred9[i]])

    df_reconstructed=pd.DataFrame(data=np.round(data), columns=column_names)
    df_reconstructed=df_reconstructed.astype(int)

  return df_reconstructed


column_names=['Pr1','Pr2','Pr3', 'FA1','FA2','FA3', 'FI1','FI2','FI3']
column_names2 =['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9']
number_list =[]
l=np.ones(shape=(1,9))
number_df=pd.DataFrame(l,columns=column_names)
clf_loaded = load('knn.joblib')
df_flag = False

choice = st.radio('DO you want to load a CSV file or to insert manually the questionary answers? :', ['CSV file', 'Manually'])
if choice=='CSV file':
  st.subheader("Load your file")
  dataframe_loaded=pd.DataFrame()
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
      dataframe_loaded = pd.read_csv(uploaded_file)
      st.write(dataframe_loaded)
  classifier=st.button("How many are Dishonest?")
  if classifier:
    y_pred = clf_loaded.predict(dataframe_loaded)
    y_pred=np.where(y_pred=='D')
    new_df=dataframe_loaded.loc[list(y_pred[0])]
    new_df['Condition']='D'
    new_df['Index']=new_df.index
    st.subheader("Classification")
    st.write(f"Founded {new_df.shape[0]} Dishonest people!")
    st.write(new_df)
    csv_downloader(new_df, "Classified")
    new_df = new_df.iloc[:,:(new_df.shape[1]-1)].to_numpy()
    st.subheader("Reconstruction")
    df_reconstructed = recon(new_df, column_names=column_names, csv=True)
    st.write(df_reconstructed)
    csv_downloader(df_reconstructed, "Reconstructed")



else:
  st.subheader("Load your answers")
  # answers=st.text_input("How many people answers do you want to write?")
  # st.write(f"answers {answers}")
  # all_answers =[]
  st.write(" Thefollowing statements reflect problem that people sometimes experience in relation to a stressful life event(s). \
  Thinking about the stressful life event(s) you identified above, please indicate how much you have been bothered by each of the following problems in the past month: ")
  df_visual = pd.DataFrame(np.ones((1,5)),columns=["Not at all", "A little bit", "Moderately", "Quite a bit", "Extremely"])
  for i in range(5):
    df_visual.iloc[:,i]=i
  st.markdown(df_visual.to_html(index=False), unsafe_allow_html=True)
  st.write("Q1: I worry a lot more since the stressful event(s)")
  number = st.number_input('Q1', min_value=0, max_value=4, step=1,  key=f'qq_1')
  number_list.append(number)

  st.write("Q2:  I cannot stop thinking about the stressful event(s)")
  number = st.number_input('Q2', min_value=0, max_value=4, step=1,  key=f'qq_2')
  number_list.append(number)

  st.write("Q3: I often feel afraid about what might happen in the future since the stressful event(s)")
  number = st.number_input('Q3', min_value=0, max_value=4, step=1,  key=f'qq_3')
  number_list.append(number)

  st.write("Q4:I find it difficult to adapt to life since the stressful event(s) ")
  number = st.number_input('Q4', min_value=0, max_value=4, step=1,  key=f'qq_4')
  number_list.append(number)
  st.write("Q5: I find it difficult to relax and feel calm since the stressful event(s)")
  number = st.number_input('Q5', min_value=0, max_value=4, step=1,  key=f'qq_5')
  number_list.append(number)
  st.write("Q6: I find it difficult to achieve a state of inner peace since the stressful event(s) ")
  number = st.number_input('Q6', min_value=0, max_value=4, step=1,  key=f'qq_6')
  number_list.append(number)
  st.write("Q7: Affected your relationships or social life?")
  number = st.number_input('Q7', min_value=0, max_value=4, step=1,  key=f'qq_7')
  number_list.append(number)
  st.write("Q8:  Affected your ability to work or your educational life? ")
  number = st.number_input('Q8', min_value=0, max_value=4, step=1,  key=f'qq_8')
  number_list.append(number)
  st.write("Q9: Affected any other important part of your life?")
  number = st.number_input('Q9', min_value=0, max_value=4, step=1,  key=f'qq_9')
  number_list.append(number)
  l=np.ones(shape=(1,9))
  number_df=pd.DataFrame(l,columns=column_names2)

  for i in range(l.shape[1]):
    number_df.iloc[:,i]= number_list[i]
  st.write(number_df)
  number_df.rename(columns={'Q1':'Pr1','Q2':'Pr2','Q3':'Pr3','Q4': 'FA1','Q5':'FA2','Q6':'FA3','Q7': 'FI1',
                    'Q8':'FI2','Q9':'FI3'}, inplace=True)

  classifier=st.button("Am I Honest or Dishonest?")
  if classifier:
    y_pred=clf_loaded.predict(number_df)
    if y_pred[0]=='H':
      st.success("You are Honest!")
    else:
      st.error("You are Dishonest")
    if y_pred[0]=='D': 
      df_reconstructed=recon(number_list,column_names2=column_names2)
      st.subheader("Reconstruction")
      st.write(df_reconstructed)
      csv_downloader(df_reconstructed, "Reconstructed_1_questionary")








    

