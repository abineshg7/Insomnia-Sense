from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import tokenizer_from_json # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import json

app = Flask(__name__)

# Define file paths
MODEL_PATH = 'insomniadm1_model.keras'
TOKENIZER_PATH = 'tokenizer.json'
TRAIN_DATA_PATH = 'TRAIN.csv'
FEEDBACK_DATA_PATH = 'feedback_data.csv'

# Check if files exist and load them
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

if not os.path.exists(TRAIN_DATA_PATH):
    raise FileNotFoundError(f"Training data file not found: {TRAIN_DATA_PATH}")

if not os.path.exists(FEEDBACK_DATA_PATH):
    # Create an empty CSV file for feedback data if it doesn't exist
    feedback_columns = ['prediction', 'feedback', 'correct_result']
    feedback_df = pd.DataFrame(columns=feedback_columns)
    feedback_df.to_csv(FEEDBACK_DATA_PATH, index=False)

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Load the tokenizer
with open(TOKENIZER_PATH, 'r') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Load and fit the scaler
train_data = pd.read_csv(TRAIN_DATA_PATH)
scaler = StandardScaler()
scaler.fit(train_data.drop(columns=['id', 'insomnia']))  # Assuming 'insomnia' is the target column

@app.route('/')
def home():
    return render_template('Home.html')  # Render home.html when accessing the root URL

@app.route('/about')
def about():
    return render_template('About.html')  # Render about.html for the /about URL

@app.route('/contact')
def contact():
    return render_template('Contact.html')  # Render contact.html for the /contact URL

@app.route('/index')
def index():
    return render_template('index.html')  # Render index.html for the /index URL

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the request contains JSON data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input, JSON data required'}), 400

    try:
        # Process structured data
        structured_data = np.array(data.get('structured_data'))
        if structured_data is None:
            return jsonify({'error': 'Structured data is missing'}), 400

        structured_data_scaled = scaler.transform(structured_data)

        # Process text data
        texts = data.get('texts')
        if not texts:
            return jsonify({'error': 'Text data is missing'}), 400

        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

        # Make predictions
        predictions = model.predict([padded_sequences, structured_data_scaled])
        predicted_classes = np.argmax(predictions, axis=1)

        # Class labels
        class_labels = ['No Insomnia', 'Acute Insomnia', 'Chronic Insomnia']
        prediction_result = class_labels[predicted_classes[0]]

        # Recommend cure based on the prediction
        cure = get_cure(prediction_result)

        return jsonify({
            'prediction': prediction_result,
            'cure': cure
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    # Ensure the request contains JSON data for feedback
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid input, JSON data required'}), 400

    try:
        # Extract feedback details from the request data
        prediction = data.get('prediction')
        feedback = data.get('feedback')
        correct_result = data.get('correct_result')

        # Save feedback to the CSV file
        feedback_df = pd.read_csv(FEEDBACK_DATA_PATH)
        new_feedback = {
            'prediction': prediction,
            'feedback': feedback,
            'correct_result': correct_result
        }
        feedback_df = feedback_df.append(new_feedback, ignore_index=True)
        feedback_df.to_csv(FEEDBACK_DATA_PATH, index=False)

        return jsonify({'status': 'Feedback saved successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_cure(prediction):
    """Return cure suggestions based on the prediction."""
    cures = {
        'No Insomnia': "Maintain a healthy sleep schedule, avoid caffeine late in the day.",
        'Acute Insomnia': "Try relaxation techniques like meditation, reduce screen time before bed.",
        'Chronic Insomnia': "Consult with a healthcare provider for sleep therapy or cognitive-behavioral treatment."
    }
    return cures.get(prediction, "No cure available.")

if __name__ == '__main__':
    app.run(debug=True)
