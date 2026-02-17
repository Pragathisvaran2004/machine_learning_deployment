import pandas as pd
import joblib
from flask import Flask, request, jsonify
import joblib
app = Flask(__name__)
model_filename = 'xgboost_weather_model.joblib'
joblib.dump(model, model_filename)

print(f"XGBoost model exported successfully to {model_filename}")

# Load the trained XGBoost model
model = joblib.load('xgboost_weather_model.joblib')

# Preprocessing artifacts from previous steps
column_means = {'u100': 1.386060554770142, 'v100': -0.6588336918310846, 'u10': 0.9796831334499894, 'v10': -0.3308165255835449, 'fg10': 6.83838937049919, 'ssrd': 820373.8757136638, 'msl': 100949.65353589428, 'sp': 51752.623034851786, 'd2m': 295.8783578191499, 't2m': 301.7398199353428, 'relative humidity': 67.34455302207745, 'cloud cover': 71.91277596815056, 'temperature': 28.668226967466925, 'wind direction 10m': 211.93139502151445, 'wind speed 10m': 10.074023806651383}
feature_columns = ['u100', 'v100', 'u10', 'v10', 'fg10', 'ssrd', 'msl', 'sp', 'd2m', 't2m', 'relative humidity', 'cloud cover', 'temperature', 'wind direction 10m', 'wind speed 10m', 'year', 'month', 'day', 'hour']

def preprocess_input_data(data):
    # Convert input data to DataFrame
    df_input = pd.DataFrame([data])

    # Handle 'time' column if present, then extract time-based features
    if 'time' in df_input.columns:
        df_input['time'] = pd.to_datetime(df_input['time'])
        df_input['year'] = df_input['time'].dt.year
        df_input['month'] = df_input['time'].dt.month
        df_input['day'] = df_input['time'].dt.day
        df_input['hour'] = df_input['time'].dt.hour
        df_input = df_input.drop(columns=['time'])
    else:
        # If 'time' is not in input, ensure year, month, day, hour are handled
        # This part assumes that if 'time' is missing, these features might be provided directly
        # or some default/error handling should be in place. For simplicity, we'll assume they're present
        # or correctly derived by the user.
        for col in ['year', 'month', 'day', 'hour']:
            if col not in df_input.columns:
                df_input[col] = 0  # Placeholder, better to raise error or ask for input

    # Ensure all feature_columns are present, fill missing numerical values with pre-calculated means
    for col in feature_columns:
        if col not in df_input.columns:
            if col in column_means:
                df_input[col] = column_means[col]
            else:
                df_input[col] = 0 # Default for new engineered columns not in means
        elif df_input[col].isnull().any():
            if col in column_means:
                df_input[col] = df_input[col].fillna(column_means[col])
            else:
                df_input[col] = df_input[col].fillna(0) # Default for other missing values

    # Reorder columns to match the training data
    df_input = df_input[feature_columns]

    return df_input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        processed_data = preprocess_input_data(data)

        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[:, 1]

        return jsonify({'precipitation_probability': prediction_proba[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For deployment, use a production-ready WSGI server like Gunicorn
    # For local testing:
    app.run(host='0.0.0.0', port=5000)
