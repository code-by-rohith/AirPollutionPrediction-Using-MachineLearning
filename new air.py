from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data and train the model
sensor_data = pd.read_csv("sensor_data.csv")
quality_data = pd.read_csv("quality_control_data.csv")

rawdataset = sensor_data.merge(quality_data, on="prod_id")
dataset = rawdataset.drop(columns='prod_id')

array = dataset.values
X = array[:,0:3]
Y = array[:,3]

# Split the dataset into training and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=8)

# Train the model
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)

# Define routes
@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        testWeight = float(request.form['weight'])
        testHumidity = float(request.form['humidity'])
        testTemperature = float(request.form['temperature'])
        
        # Make prediction
        testPrediction = CART.predict([[testWeight, testHumidity, testTemperature]])
        
        return render_template('index.html', prediction_text=f"Air Quality: {testPrediction}")

if __name__ == '__main__':
    app.run(debug=True)
