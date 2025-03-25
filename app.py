from flask import Flask, render_template, request
import pickle
import numpy as np
import json
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained KNN model and column data mappings
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('column_data.json', 'r') as f:
    column_data = json.load(f)

# Get the feature names from the model
column_names = knn_model.feature_names_in_

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        Age = int(request.form['Age'])
        Gender = request.form['Gender']
        AvgPurchaseAmount_INR = float(request.form['AvgPurchaseAmount (INR)'])
        Segment = request.form['Segment']
        PurchaseHistory = request.form['PurchaseHistory']
        PurchaseFrequency = request.form['PurchaseFrequency']
        date_difference = int(request.form['date_difference'])

        # Transform categorical variables using column_data mappings
        Gender = column_data['Gender'][Gender]
        Segment = column_data['Segment'][Segment]

        # Create binary variables for the PurchaseHistory
        PurchaseFrequency_Monthly = 1 if PurchaseFrequency == 'Monthly' else 0
        PurchaseFrequency_Quarterly = 1 if PurchaseFrequency == 'Quarterly' else 0

        Footwear = 1 if PurchaseHistory == 'Footwear' else 0
        Casual_Wear = 1 if PurchaseHistory == 'Casual_Wear' else 0
        Traditional_Wear = 1 if PurchaseHistory == 'Traditional_Wear' else 0
        Accessories = 1 if PurchaseHistory == 'Accessories' else 0
        Gym_Wear = 1 if PurchaseHistory == 'Gym Wear' else 0

        # Prepare the feature array (test data)
        test_array = np.zeros((1, len(column_names)))

        test_array[0, column_names.tolist().index('Age')] = Age
        test_array[0, column_names.tolist().index('Gender')] = Gender
        test_array[0, column_names.tolist().index('AvgPurchaseAmount (INR)')] = AvgPurchaseAmount_INR
        test_array[0, column_names.tolist().index('Segment')] = Segment
        test_array[0, column_names.tolist().index('Footwear')] = Footwear
        test_array[0, column_names.tolist().index('Casual_Wear')] = Casual_Wear
        test_array[0, column_names.tolist().index('Traditional_Wear')] = Traditional_Wear
        test_array[0, column_names.tolist().index('Accessories')] = Accessories
        test_array[0, column_names.tolist().index('Gym Wear')] = Gym_Wear
        test_array[0, column_names.tolist().index('date_difference')] = date_difference
        test_array[0, column_names.tolist().index('PurchaseFrequency_Monthly')] = PurchaseFrequency_Monthly
        test_array[0, column_names.tolist().index('PurchaseFrequency_Quarterly')] = PurchaseFrequency_Quarterly

        # Convert the test array to a DataFrame with proper column names
        test_df = pd.DataFrame(test_array, columns=column_names)

        # Make prediction
        prediction = knn_model.predict(test_df)[0]

        # Render the result and return the form with the filled data
        return render_template('index.html', 
                               prediction=prediction,
                               Age=Age, 
                               Gender=Gender, 
                               AvgPurchaseAmount_INR=AvgPurchaseAmount_INR, 
                               Segment=Segment,
                               PurchaseHistory=PurchaseHistory, 
                               PurchaseFrequency=PurchaseFrequency, 
                               date_difference=date_difference)

if __name__ == '__main__':
    app.run(debug=True)
