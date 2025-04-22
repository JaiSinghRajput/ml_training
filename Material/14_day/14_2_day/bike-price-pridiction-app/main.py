from flask import Flask, render_template,request
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
data = pd.read_csv(r"D:\CLG TRAINING\Material\09_day\Used_Bikes.csv")
model = pickle.load(open(r"D:\CLG TRAINING\Material\14_day\bike-price-pridiction-app\model.pkl", "rb"))
brand = data["brand"].unique()
brand = np.sort(brand)
brand = brand.tolist()
owner = data["owner"].unique()
owner = np.sort(owner)
owner = owner.tolist()
@app.route("/")
def index():
    return render_template("index.html", brands=brand,owners=owner)
@app.route("/about")
def about():
    return render_template("about.html")
# api routes
@app.route("/api/getPrediction", methods=["POST"])
def getPrediction():
    form_data = request.get_json()
    print("Received data:", form_data)

    try:
        # Extract fields from request
        kms_driven = float(form_data["kms_driven"])
        owner = form_data["owner"]
        age = float(form_data["age"])
        power = float(form_data["power"])
        brand = form_data["brand"]

        # Put them in the correct order expected by the pipeline
        input_data = pd.DataFrame([{
            "kms_driven": kms_driven,
            "owner": owner,
            "age": age,
            "power": power,
            "brand": brand
        }])

        # Make prediction using the pipeline
        price = model.predict(input_data)[0]
        price = round(price, 2)

        return {"success": True, "price": price}

    except Exception as e:
        print("Prediction error:", str(e))
        return {"success": False, "message": str(e)}
if __name__ == '__main__':
    app.run(debug=True)
    # 22500.0	First Owner	6.0	350.0	Royal Enfield