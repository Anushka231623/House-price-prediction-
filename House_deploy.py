import pickle
import numpy as np
import streamlit as st

# Load the model
model_path = r"C:\Users\Personal\trained_model.sav"  # Update this with the correct path

try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

def predict_price(entries):
    try:
        # Get user input
        input_data = [int(entries[0]), float(entries[1]), int(entries[2]), int(entries[3]), 
                      float(entries[4]), int(entries[5]), int(entries[6]), int(entries[7]), 
                      int(entries[8]), int(entries[9]), int(entries[10]), int(entries[11])]
        
        # Perform prediction
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        predicted_price = loaded_model.predict(input_data_reshaped)[0]
        return f"The predicted price is ${predicted_price:,.2f}"
    except ValueError:
        return "Please enter valid inputs."

def main():
    st.title("House Price Prediction")
    entries = []
    for feature in ['Bedrooms:', 'Bathrooms:', 'Sqft Living:', 'Sqft Lot:', 'Floors:', 
                    'Waterfront:', 'View:', 'Condition:', 'Sqft Above:', 'Sqft Basement:', 
                    'Year Built:', 'Year Renovated:']:
        entries.append(st.number_input(feature))
    
    if st.button('Predict Price'):
        result = predict_price(entries)
        st.success(result)

if __name__ == '__main__':
    main()
