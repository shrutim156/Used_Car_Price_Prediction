from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)

# Load model with error checking
try:
    pipeline = joblib.load('final_pipeline.joblib')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model failed to load: {e}")
    pipeline = None


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Collect form data into variables directly
    model_name        = request.form.get('model')
    vehicle_age       = int(request.form.get('vehicle_age'))
    km_driven         = float(request.form.get('km_driven'))
    mileage           = float(request.form.get('mileage'))
    engine            = float(request.form.get('engine'))
    max_power         = float(request.form.get('max_power'))
    seats             = int(request.form.get('seats'))
    fuel_type         = request.form.get('fuel_type')
    transmission_type = request.form.get('transmission_type')
    seller_type       = request.form.get('seller_type')

    # Build dataframe for prediction
    input_df = pd.DataFrame([{
        'model':             model_name,
        'vehicle_age':       vehicle_age,
        'km_driven':         km_driven,
        'mileage':           mileage,
        'engine':            engine,
        'max_power':         max_power,
        'seats':             seats,
        'fuel_type':         fuel_type,
        'transmission_type': transmission_type,
        'seller_type':       seller_type
    }])

    prediction   = pipeline.predict(input_df)
    actual_price = np.exp(prediction[0])
    prediction_text = f"Predicted Price: ₹{actual_price:,.0f}"

    return render_template('result.html',
        prediction_text   = prediction_text,
        model             = model_name,       # ← model_name, not model
        vehicle_age       = vehicle_age,
        km_driven         = km_driven,
        mileage           = mileage,
        engine            = engine,
        max_power         = max_power,
        seats             = seats,
        fuel_type         = fuel_type,
        transmission_type = transmission_type,
        seller_type       = seller_type
    )


if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)

