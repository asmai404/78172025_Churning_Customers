import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import pickle

encoding = {"Yes": 1,
            "No": 0,
            "Female": 0,
            "Male": 1,
            "No phone service": 2,
            "Bank transfer (automatic)": 0,
            "Electronic check": 2,
            "Mailed check": 3,
            "Credit card (automatic)": 1,
            "DSL": 0,
            "Fiber optic": 1,
            "Month-to-month": 0,
            "One-year": 1,
            "Two-year": 2
            }

app = Flask(__name__)


# Loading trained model from a pickle file
# with open('churn_pred.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
#

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        gender = encoding[request.form['gender']]
        senior_citizen = int(request.form['senior_citizen'])
        partner = encoding[request.form['partner']]
        dependents = encoding[request.form['dependents']]
        tenure = int(request.form['tenure'])
        phone_service = encoding[request.form['phone_service']]
        multiple_lines = encoding[request.form['multiple_lines']]
        internet_service = encoding[request.form['internet_service']]
        online_security = encoding[request.form['online_security']]
        online_backup = encoding[request.form['online_backup']]
        device_protection = encoding[request.form['device_protection']]
        tech_support = encoding[request.form['tech_support']]
        streaming_tv = encoding[request.form['streaming_tv']]
        streaming_movies = encoding[request.form['streaming_movies']]
        contract = encoding[request.form['contract']]
        paperless_billing = encoding[request.form['paperless_billing']]
        payment_method = encoding[request.form['payment_method']]
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])

        # Create a feature vector with the input data
        input_data = {
            "gender": [gender],
            "senior_citizen": [senior_citizen],
            "partner": [partner],
            "dependents": [dependents],
            "tenure": [tenure],
            "phone_service": [phone_service],
            "multiple_lines": [multiple_lines],
            "internet_service": [internet_service],
            "online_security": [online_security],
            "online_backup": [online_backup],
            "device_protection": [device_protection],
            "tech_support": [tech_support],
            "streaming_tv": [streaming_tv],
            "streaming_movies": [streaming_movies],
            "contract": [contract],
            "paperless_billing": [paperless_billing],
            "payment_method": [payment_method],
            "monthly_charges": [monthly_charges],
            "total_charges": [total_charges]
        }

        # print(list(input_data.values()))
        # sc = StandardScaler()
        # scaled = sc.fit_transform(list(input_data.values()))
        # print(scaled)

        input_data_scaled = pd.DataFrame(input_data, columns=input_data.keys())

        # Make predictions using your model

        result = model.predict(input_data_scaled)

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
