from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load and preprocess the dataset
try:
    dataframe = pd.read_csv('dataset.csv')
    dataframe.drop('education', axis=1, inplace=True)
    dataframe.rename(columns={"TenYearCHD": "CHD"}, inplace=True)
except FileNotFoundError:
    print("Error: dataset.csv not found. Please ensure the file is in the same directory as app.py.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Split features and target
try:
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
except Exception as e:
    print(f"Error splitting data: {str(e)}")
    exit(1)

# Train the model
try:
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='most_frequent')
    cols_to_standardise = ['age', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose', 'cigsPerDay']
    X[cols_to_standardise] = scaler.fit_transform(X[cols_to_standardise])
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    model = RandomForestClassifier(n_estimators=100, max_depth=None, criterion='gini')
    model.fit(X, y)
except Exception as e:
    print(f"Error training model: {str(e)}")
    exit(1)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error rendering template: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'male': int(request.form['male']),
            'age': float(request.form['age']),
            'currentSmoker': int(request.form['currentSmoker']),
            'cigsPerDay': float(request.form['cigsPerDay']),
            'BPMeds': int(request.form['BPMeds']),
            'prevalentStroke': int(request.form['prevalentStroke']),
            'prevalentHyp': int(request.form['prevalentHyp']),
            'diabetes': int(request.form['diabetes']),
            'totChol': float(request.form['totChol']),
            'sysBP': float(request.form['sysBP']),
            'diaBP': float(request.form['diaBP']),
            'BMI': float(request.form['BMI']),
            'heartRate': float(request.form['heartRate']),
            'glucose': float(request.form['glucose'])
        }

        # Create DataFrame from input
        input_df = pd.DataFrame([data], columns=X.columns)

        # Preprocess input
        input_df[cols_to_standardise] = scaler.transform(input_df[cols_to_standardise])
        input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        # Return result
        result = {
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'probability': round(probability, 2)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)