import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

# Initialize Flask
application = Flask(__name__)
app=application

#import my model and scaler
esk_model=pickle.load(open('models/esk.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# Define a route for home page

@app.route('/prediction', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH          = float(request.form.get('RH'))
        Ws          = float(request.form.get('Ws'))
        Rain        = float(request.form.get('Rain'))
        FFMC        = float(request.form.get('FFMC'))
        DMC         = float(request.form.get('DMC'))
        ISI         = float(request.form.get('ISI'))
        Classes     = float(request.form.get('Classes'))
        Region      = float(request.form.get('Region'))
        
        # Raw input
        features = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        print("🔹 Raw input:", features)

        # After scaling
        new_data = standard_scaler.transform(features)
        print("🔹 Scaled input:", new_data)

        # Prediction
        result   = esk_model.predict(new_data)[0]
        print("🔹 Raw prediction:", result[0])

        return render_template('home.html', prediction_text=f"Predicted FWI: {result}")
    
    return render_template('predict.html')

  

# Run the app
if __name__ == "__main__":
    app.run()
