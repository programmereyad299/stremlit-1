import streamlit as st
import joblib
import pandas as pd

#load
model = joblib.load("model (1).pkl")
y = joblib.load("y.pkl")
le1 = joblib.load("le1 (1).pkl")
le4 = joblib.load("le4.pkl")

Time_spent_Alone = st.number_input("Time_spent_Alone" , step= 1.0 ,min_value = 0.0 , max_value = 12.0)

Stage_fear = st.selectbox("Stage_fear", ["Yes" , "No"])

Social_event_attendance = st.number_input("Social_event_attendance " , step= 1.0 ,min_value = 0.0 , max_value = 10.0)
Going_outside =  st.number_input("Going_outside " , step= 1.0 ,min_value = 0.0 , max_value = 8.0)

Drained_after_socializing =st.selectbox("Drained_after_socializing ", ["Yes" , "No"])

Friends_circle_size = st.number_input("Friends_circle_size " , step= 1.0 ,min_value = 0.0 , max_value = 15.0)
Post_frequency =st.number_input("Post_frequency " , step= 1.0 ,min_value = 0.0 , max_value = 10.0)


Stage_fear_encode = le1.transform([Stage_fear])[0]
Drained_after_socializing_encode = le4.transform([Drained_after_socializing ])[0]

input_df = pd.DataFrame([[Time_spent_Alone  ,Stage_fear_encode ,Social_event_attendance,Going_outside ,Drained_after_socializing_encode ,Friends_circle_size ,Post_frequency]] ,
                         columns=["Time_spent_Alone"  ,"Stage_fear" ,"Social_event_attendance","Going_outside" ,"Drained_after_socializing","Friends_circle_size" ,"Post_frequency"])

pred = model.predict(input_df)[0]

btn = st.button("pred")
if btn:
    st.write(y.inverse_transform([pred])[0])

