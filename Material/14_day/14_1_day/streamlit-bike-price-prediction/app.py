import pandas as pd
import pickle
import streamlit as st
data = pd.read_csv(r"D:\CLG TRAINING\Material\09_day\Used_Bikes.csv")
pipe = pickle.load(open(r"D:\CLG TRAINING\Material\14_day\14_1_day\bike-price-pridiction-app\model.pkl", "rb"))
history = pd.read_csv(r"D:\CLG TRAINING\Material\14_day\14_1_day\bike-price-pridiction-app\history.csv")
history.dropna(inplace=True)
history.drop_duplicates(inplace=True)
st.title("Bike Price Prediction")
st.write("This app predicts the price of used bikes based on various features.")
st.sidebar.header("Select the Brand of Bike")
brand = st.sidebar.selectbox("Choose a brand:",data["brand"].unique())
power = st.selectbox("Choose the CC Power of your bike : ",data["power"].unique())
age = st.number_input("Enter how old Your Bike: ",0,20)
kms_driven = st.slider("select the Kilometer driven of your bike",1,99999,1000,1000)
owner = st.radio("Select the Owner :",data["owner"].unique())
input_data = pd.DataFrame([{"kms_driven": kms_driven,"owner": owner,"age": age,"power": power,"brand": brand}])
predict = st.button("Get Bike Price ")
if predict:
    price = pipe.predict(input_data)[0]
    price = round(price, 2)
    input_data["date"] = pd.to_datetime("today").date()
    input_data["predicted_price"] = price
    st.success(f"Your Bike Price is : {price}")
    history = pd.concat([history,input_data], ignore_index=True)
    history.to_csv(r"D:\CLG TRAINING\Material\14_day\14_1_day\bike-price-pridiction-app\history.csv",index=False)
    st.write("Data saved successfully!")
col1,col2 = st.columns(2)
with col1:
    st.subheader("Pridiction History")
    st.dataframe(history, use_container_width=True)
with col2:
    st.scatter_chart(history["brand"], use_container_width=True, height=300)
    st.write("Brand distribution of predicted bike prices.")