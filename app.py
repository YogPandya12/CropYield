from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Flask app

app = Flask(__name__)

# Load preprocessor and model
with open('artifacts/CropPreprocessor.pkl', 'rb') as preprocessor_file:
    CropPreprocessor = pickle.load(preprocessor_file)

with open('artifacts/CropModel.pkl', 'rb') as file:
    CropModel = pickle.load(file)

with open('artifacts/CropPricePreprocessor.pkl', 'rb') as preprocessor_file:
    CropPricePreprocessor = pickle.load(preprocessor_file)

with open('artifacts/CropPriceModel.pkl', 'rb') as file:
    CropPriceModel = pickle.load(file)

Label_Mapping= {'Amaranthus': 0, 'Amla': 1, 'Amphophalus': 2, 'Apple': 3, 'Arecanut': 4, 'Ash Gourd': 5, 'Bajra': 6, 'Banana': 7, 'Barley': 8, 'Beans': 9, 'Beetroot': 10, 'Bengal Gram': 11, 'Betal Leaves': 12, 'Bitter Gourd': 13, 'Black Gram': 14, 'Black pepper': 15, 'Bottle Gourd': 16, 'Brinjal': 17, 'Broken Rice': 18, 'Cabbage': 19, 'Capsicum': 20, 'Carrot': 21, 'Cashewnuts': 22, 'Castor Seed': 23, 'Cauliflower': 24, 'Chana Dal': 25, 'Chholia': 26, 'Chilly Capsicum': 27, 'Cluster Beans': 28, 'Cluster beans': 29, 'Coconut': 30, 'Coconut Oil': 31, 'Coconut Seed': 32, 'Colacasia': 33, 'Copra': 34, 'Coriander': 35, 'Corriander seed': 36, 'Cotton': 37, 'Cowpea': 38, 'Cucumber': 39, 'Cumbu': 40, 'Drumstick': 41, 'Dry Chillies': 42, 'Duster Beans': 43, 'Elephat Yam': 44, 'Field Pea': 45, 'Fish': 46, 'French Beans': 47, 'Garlic': 48, 'Ghee': 49, 'Gingelly Oil': 50, 'Ginger': 51, 'Grapes': 52, 'Green Avare': 53, 'Green Banana': 54, 'Green Chilli': 55, 'Green Gram': 56, 'Green Onion': 57, 'Green Peas': 58, 'Ground Nut Seed': 59, 'Groundnut': 60, 'Guar': 61, 'Guava': 62, 'Horse Gram': 63, 'Hybrid Cumbu': 64, 'Jaggery': 65, 'Jowar': 66, 'Jute': 67, 'Karamani': 68, 'Kinnow': 69, 'Knool Khol': 70, 'Ladies Finger': 71, 'Leafy Vegetable': 72, 'Lemon': 73, 'Lentil': 74, 'Lime': 75, 'Linseed': 76, 'Little Gourd': 77, 'Long Melon': 78, 'Maida Atta': 79, 'Maize': 80, 'Mango': 81, 'Masur Dal': 82, 'Methi Leaves': 83, 'Mint': 84, 'Moath Dal': 85, 'Moong Dal': 86, 'Mushrooms': 87, 'Musk Melon': 88, 'Mustard': 89, 'Mustard Oil': 90, 'Niger Seed': 91, 'Onion': 92, 'Orange': 93, 'Paddy': 94, 'Papaya': 95, 'Parval': 96, 'Pear': 97, 'Peas': 98, 'Peas cod': 99, 'Pepper garbled': 100, 'Pigeon Pea': 101, 'Pineapple': 102, 'Plum': 103, 'Pomegranate': 104, 'Potato': 105, 'Pumpkin': 106, 'Raddish': 107, 'Ragi': 108, 'Rajgir': 109, 'Red Gram': 110, 'Rice': 111, 'Ridge Gourd': 112, 'Round gourd': 113, 'Rubber': 114, 'Sapota': 115, 'Seam': 116, 'Seemebadnekai': 117, 'Sesamum': 118, 'Snakeguard': 119, 'Soyabean': 120, 'Spinach': 121, 'Sponge Gourd': 122, 'Squash': 123, 'Sugar': 124, 'Surat Beans': 125, 'Suvarna Gadde': 126, 'Sweet Lime': 127, 'Sweet Potato': 128, 'Sweet Pumpkin': 129, 'Tamarind Fruit': 130, 'Tapioca': 131, 'Taramira': 132, 'Tender Coconut': 133, 'Thinai': 134, 'Tinda': 135, 'Tobacco': 136, 'Tomato': 137, 'Tur Dal': 138, 'Turmeric': 139, 'Turnip': 140, 'Urd Dal': 141, 'Varagu': 142, 'Water Melon': 143, 'Wheat': 144, 'Wheat Atta': 145, 'White Pumpkin': 146, 'Wood': 147, 'Zizyphus': 148}
def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')



@app.route("/predict", methods=["GET",'POST'])
def predict():
    if request.method== 'POST':
        # Retrieve input data from the form and convert to appropriate types
        rainfall = request.form['Rainfall']
        humidity = request.form['Humidity']
        state = request.form['states']
        temperature = request.form['temperature']
        phlevel = request.form['phLevel']  # Match this with HTML
        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        soiltype = request.form['soilType']  # Match this with HTML

        # Create a feature array with actual values
        features = np.array([[state, soiltype, nitrogen, phosphorous, potassium, temperature, humidity, phlevel, rainfall]])

        # Define column names
        column_names = ['STATE', 'SOIL_TYPE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']

        # Create DataFrame with input data
        input_data_df1 = pd.DataFrame(features, columns=column_names)

        # Transform the input data
        transformed_features = CropPreprocessor.transform(input_data_df1)

        # Make prediction
        CropPrediction_No = CropModel.predict(transformed_features)
        CropPrediction = get_key_by_value(Label_Mapping, CropPrediction_No)

        features=np.append(features,CropPrediction)
        features = features.reshape(1, -1)
        column_names.append('CROP')
        # Create DataFrame with input data
        input_data_df2 = pd.DataFrame(features, columns=column_names)

        # Transform the input data
        transformed_features = CropPricePreprocessor.transform(input_data_df2)

        # Make prediction
        CropPricePrediction = round(CropPriceModel.predict(transformed_features)[0],2)

        ans={
            'Crop':CropPrediction,
            'CropPrice':CropPricePrediction
        }
        # Render the result on the prediction page
        return render_template('result.html', prediction=ans['Crop'],pricePrediction=ans['CropPrice'])
    return render_template('crop_yield.html')


if __name__ == "__main__":
    app.run(debug=True)
