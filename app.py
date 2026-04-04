from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)

#load your saved model
model = joblib.load('final_pipeline.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    data = {
        'model': request.form.get('model'),
        'vehicle_age': int(request.form.get('vehicle_age')),
        'km_driven': float(request.form.get('km_driven')),
        'mileage': float(request.form.get('mileage')),
        'engine': float(request.form.get('engine')),
        'max_power': float(request.form.get('max_power')),
        'seats': int(request.form.get('seats')),
        'fuel_type': request.form.get('fuel_type'),
        'transmission_type': request.form.get('transmission_type'),
        'seller_type': request.form.get('seller_type')
        }
    
    input_df = pd.DataFrame([data]) 

    prediction = model.predict(input_df)

    actual_price = np.exp(prediction[0])  # reverses log transform

    return render_template("index.html", prediction_text=f"Predicted Price: ₹{actual_price:,.0f}")

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)

