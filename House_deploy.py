import streamlit as st
import requests
import joblib

def load_model():
    model_url = "https://github.com/Anushka231623/House-price-prediction-/raw/main/finalized_model.sav"
    r = requests.get(model_url)

    if r.status_code == 200:
        with open('finalized_model.sav', 'wb') as f:
            f.write(r.content)
    else:
        print("Failed to download the model file")

def load_and_predict(entries):
    try:
        model = joblib.load('finalized_model.sav')

        input_data = [int(entries[0]), float(entries[1]), int(entries[2]), int(entries[3]), 
                      float(entries[4]), int(entries[5]), int(entries[6]), int(entries[7]), 
                      int(entries[8]), int(entries[9]), int(entries[10]), int(entries[11])]

        print("Input data:", input_data)  # Print input data for debugging
        
        if len(input_data) != 12:
            return "Failed to predict price: Invalid input data length"
        
        input_data = [input_data]  # Reshape input data
        prediction = model.predict(input_data)
        return f"The predicted price is ${prediction[0]:,.2f}"
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"Failed to predict price: {e}"  # Return the specific error message

def main():
    st.title("House Price Prediction")
    entries = []
    for feature in ['Bedrooms:', 'Bathrooms:', 'Sqft Living:', 'Sqft Lot:', 'Floors:', 
                    'Waterfront:', 'View:', 'Condition:', 'Sqft Above:', 'Sqft Basement:', 
                    'Year Built:', 'Year Renovated:']:
        entries.append(st.number_input(feature))
    
    if st.button('Predict Price'):
        load_model()  # Ensure model is loaded before making predictions
        result = load_and_predict(entries)
        st.success(result)

if __name__ == '__main__':
    main()

# Evaluate the model separately
try:
    model = joblib.load('finalized_model.sav')

    # Example input data for prediction
    input_data = [3, 2.5, 2000, 5000, 2, 0, 1, 3, 1800, 200, 1990, 2010]

    # Reshape input data
    input_data = [input_data]

    # Make prediction
    prediction = model.predict(input_data)

    print(f"The predicted price is ${prediction[0]:,.2f}")
except Exception as e:
    print(f"An error occurred during separate prediction: {e}")
