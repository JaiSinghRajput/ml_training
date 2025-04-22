from flask import Flask, render_template,request
import pandas as pd 
import pickle

# Load the model
pipe = pickle.load(open(r'D:\CLG TRAINING\Material\09_day\model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/history1')
def history1():
    return render_template('history1.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/predict', methods=['POST'])
def predict():
    brand = request.form.get('brand_name')
    owner = request.form.get('owner')
    age = request.form.get('age')
    power = request.form.get('power')
    kms_driven = request.form.get('kms_driven')
    input_data = pd.DataFrame([{
            "kms_driven": kms_driven,
            "owner": owner,
            "age": age,
            "power": power,
            "brand": brand
        }])
    # Make prediction using the pipeline
    price = pipe.predict(input_data)[0]
    price = round(price, 2)
    # Return the prediction result
    return render_template('project.html', prediction=price)

if __name__ == '__main__':
    app.run(debug=True) 