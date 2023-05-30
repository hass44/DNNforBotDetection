from flask import Flask, render_template, request
from keras.models import load_model
import pandas as pd
import numpy as np
import getaccount
import NLP_Project_Bot
import os

model = load_model('D:/DNNforBotDetection/DNNforBotDetection/models/model.h5')

app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the user input from the form
    twitter_account = request.form['twitter_account']
    
    # Fetch user data from the provided Twitter account
    df = getaccount.getUserData(twitter_account)
    
    os.remove("D:/DNNforBotDetection/DNNforBotDetection/src/twitter_api_data.json")
    # Drop unnecessary columns
    
    df = NLP_Project_Bot.clean(df)
    # Perform the prediction
    y_pred_prob = model.predict(df)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    print(y_pred_prob)
    # Determine the result message

    for pred in y_pred:
        if pred == 1:
            result = 'Not Bot'
        elif y_pred == 0:
            result = 'Bot'
    
    # Render the result template
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)