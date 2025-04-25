from flask import Flask, render_template, request
import pickle
pipeline = pickle.load(open("./static/crop_pipeline.pkl", "rb"))
label_encoder = pickle.load(open("./static/label_encoder.pkl", "rb"))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend')
def recommend():
    return render_template('recommand.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = pipeline.predict(input_data)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        return render_template('result.html', crop=f'Crop: {prediction_label}')
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/contact")
def contact():
    return render_template("contact.html")
if __name__ == "__main__":
    app.run(debug=True)
