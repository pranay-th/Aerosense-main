from flask import Flask, render_template, request, jsonify, g
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
import cv2
import pickle
import os
import requests
from bs4 import BeautifulSoup
from datetime import date
import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import csv

app = Flask(__name__, static_folder='static', template_folder='templates')

# Ensure static/uploads directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Allowed image extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize the InferenceHTTPClient with Roboflow API details
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="PVwuzN6zGNqWT1UNwvUl"
)

def train_models():
    # Load dataset for classification
    dataset_class = pd.read_csv("weather_data.csv")
    x_class = dataset_class[['Day', 'Month', 'Year']]
    y_class = dataset_class['heat']
    z_class = dataset_class['wet']

    # Train classifiers
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(x_class, y_class)

    classifier_z = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier_z.fit(x_class, z_class)


    return classifier, classifier_z

classifier, classifier_z = train_models()

# Check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the damage prevention model safely
def load_model():
    model_path = os.path.join('sub-pages', 'damage_prevention', 'models', 'model1.pkl')
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

@app.before_request
def load_prevention_model():
    """Load the model and store it in Flask.g for each request to ensure thread safety."""
    if 'model' not in g:
        g.model = load_model()

# Weather scraping function
def get_weather():
    weather = []
    year = date.today().year
    url = f"http://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{year}08.xml"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')
    
    body = soup.find("body").text.split(",")
    day = body[-27][4:-1].lstrip("0")
    weather.append(day)
    
    weather.append(body[0][-1])  # Month
    weather.append(year)  # Year
    weather.append(body[-24][1:-1])  # Mean Temp
    weather.append(body[-25][1:-1])  # Max Temp
    weather.append(body[-23][1:-1])  # Min Temp
    weather.append(body[-21][1:-1])  # Humidity
    weather.append(body[-22][1:-1])  # Dew Point
    weather.append(body[-26][1:-1])  # Pressure
    
    heat = "YES" if float(body[-24][1:-1]) >= 30 else "NO"
    wet = "YES" if float(body[-21][1:-1]) >= 80 else "NO"
    weather.extend([heat, wet])
    
    return weather

# Store weather data into CSV
def save_weather_to_csv():
    weather_data = get_weather()
    filename = 'weather_data.csv'
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(weather_data)
    return weather_data
   

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

# Prediction route (for damage prevention model)
@app.route('/predict', methods=['POST'])
def predict_prevention():
    # Collecting 14 input features from the form
    features = [float(request.form[f'feature{i+1}']) for i in range(14)]
    
    # Making predictions using the model
    prediction = g.model.predict([features])
    
    # Returning the prediction to the HTML page
    return render_template('prevention.html', prediction_text='Prediction: {}'.format(prediction[0]))

@app.route('/damage-detection')
def damage_detection():
    return render_template('damage_detection.html')

@app.route('/damage-detection', methods=['POST'])
def detect_damage():
    try:
        # Check if the POST request has the file part
        if 'image' not in request.files:
            return render_template('damage_detection.html', error="No file part")

        image = request.files['image']
        if image.filename == '':
            return render_template('damage_detection.html', error="No selected file")

        if not allowed_file(image.filename):
            return render_template('damage_detection.html', error="Only image files (JPG, PNG) are allowed")

        # Secure the filename and save the image
        filename = secure_filename(image.filename)
        image_path = os.path.join('static/uploads', filename)
        image.save(image_path)

        # Perform inference using Roboflow
        result = CLIENT.infer(image_path, model_id="aircraft-damage-detection-2/3")
        if not result or 'image' not in result or 'predictions' not in result:
            return render_template('damage_detection.html', error="Invalid response from inference client")

        detection_result = {
            'image_dimensions': f"Width: {result['image']['width']}, Height: {result['image']['height']}",
            'description': result['predictions']
        }

        return render_template('damage_detection.html', result=detection_result)

    except Exception as e:
        return render_template('damage_detection.html', error=f"Error occurred: {str(e)}")



@app.route('/object-detection')
def object_detection():
    return render_template('object_detection.html')

# Object Detection inference (handling image upload)
@app.route('/object-detection', methods=['POST'])
def detect_objects():
    image = request.files['image']
    image_path = os.path.join('static/uploads', image.filename)
    image.save(image_path)

    # Perform inference using Roboflow
    result = CLIENT.infer(image_path, model_id="fod-ia5dn/1")  # Replace with your model ID

    # Load the image with OpenCV
    img = cv2.imread(image_path)

    # Draw bounding boxes on the image
    for pred in result.get('predictions', []):
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        confidence = pred['confidence']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the image with bounding boxes
    result_image_path = os.path.join('static/uploads', 'result_' + image.filename)
    cv2.imwrite(result_image_path, img)

    # Prepare the result to be passed to the template
    detection_result = {
        'image_url': result_image_path,
        'predictions': result.get('predictions', [])
    }

    # Render the result in the HTML template
    return render_template('object_detection.html', result=detection_result)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'day' in request.form:
            # Handle the first prediction model
            day = int(request.form['day'])
            month = int(request.form['month'])
            year = int(request.form['year']) - 1999

            heat_pred = classifier.predict([[day, month, year]])
            wet_pred = classifier_z.predict([[day, month, year]])

            heat_status = "YES" if (heat_pred[0]) == 1 else "NO"
            wet_status = "YES" if (wet_pred[0]) == 1 else "NO"

            return jsonify({'heat': heat_status, 'wet': wet_status})

    return render_template('predict.html')

@app.route('/scrape', methods=['GET'])
def scrape_weather():
    # Call the weather scraping function
    weather_data = save_weather_to_csv()
    return jsonify({'message': 'Weather data scraped and saved successfully!', 'data': weather_data})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)
