import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load preprocessor and model
with open('artifacts/CropPricePreprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('artifacts/CropPriceModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Example feature values for a single sample
input_data = np.array([['Gujarat', 'Sandy soil', 0, 101, 220, 430, 560, 6.5, 0 , 'Black pepper']])  # Replace with actual values

# Column names should match those used in your model
column_names = ['STATE', 'SOIL_TYPE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE',
                'HUMIDITY', 'ph', 'RAINFALL','CROP']

# Create DataFrame with the input data
input_data_df = pd.DataFrame(input_data, columns=column_names)

# Transform the input data
processed_data = preprocessor.transform(input_data_df)

# Make predictions
predictions = model.predict(processed_data)

# Output the predictions
print("Predictions:", predictions)
