from flask import Flask, render_template, request
import joblib

app=Flask(__name__)

#load your saved model
model = joblib.load('final_pipeline.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    return "prediction logic goes here!"

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)

