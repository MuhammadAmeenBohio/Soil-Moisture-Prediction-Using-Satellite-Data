from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
from PIL import Image
import io
import os
import h5py
import matplotlib.pyplot as plt
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)  # Set default logging level to ERROR
logger = logging.getLogger(__name__)

# Load the pre-trained models
MODEL_DIRECTORY = 'E:/PRACTICE/python/pythonProject/models/'
MODEL_INPUT_SIZE = (10, 10) # Define the expected input size here
try:
    rf_model = joblib.load(os.path.join(MODEL_DIRECTORY, 'rf_model.pkl'))
    cat_model = joblib.load(os.path.join(MODEL_DIRECTORY, 'cat_model.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIRECTORY, 'xgb_model.pkl'))
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit(1) # Exit if models can't load


# Constants
MODEL_INPUT_SIZE = (10, 10)
HEATMAP_SIZE = (60, 30)  # New heatmap size (inches)


def preprocess_image(image):
    try:
        image = Image.open(io.BytesIO(image))
        image = image.resize(MODEL_INPUT_SIZE)
        image_array = np.array(image) / 255.0
        # Ensure the image has only one channel if your model expects that
        if image_array.ndim == 3 and image_array.shape[2] == 3: #check if RGB
            image_array = image_array.mean(axis=2) # Convert to grayscale
        return image_array.reshape(1, -1)  # Reshape to (1, 100) if grayscale
    except Exception as e:
        logger.exception(f"Error processing image: {e}") # Use exception to log the traceback
        raise

def create_heatmap_from_h5(h5_filename='processed_image.h5', output_filename='heatmap.png'):
    try:
        with h5py.File(h5_filename, 'r') as h5f:
            image_data = h5f['image_data'][:]

        reshaped_image = image_data.reshape(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], -1)

        fig, axes = plt.subplots(1, reshaped_image.shape[2], figsize=HEATMAP_SIZE)

        channel_names = ['Red', 'Green', 'Blue']  # Or get this from your image data somehow

        for i in range(reshaped_image.shape[2]):
            ax = axes[i] if reshaped_image.shape[2] > 1 else axes
            im = ax.imshow(reshaped_image[:, :, i], cmap='hot', interpolation='nearest')

            # Increased title font size
            ax.set_title(f'{channel_names[i]} Channel', fontsize=16) # Adjust font size as needed
            ax.axis('off')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['0 (Dry)', '0.5', '1 (Moist)'])


        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

    except Exception as e:
        logging.exception(f"Error creating heatmap: {e}")
        raise



def create_combined_heatmap_from_h5(h5_filename='processed_image.h5', output_filename='combined_heatmap.png'):
    try:
        with h5py.File(h5_filename, 'r') as h5f:
            image_data = h5f['image_data'][:]

        reshaped_image = image_data.reshape(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], -1)

        fig, axes = plt.subplots(1, reshaped_image.shape[2], figsize=HEATMAP_SIZE)

        channel_names = ['Red', 'Green', 'Blue']  # Names for the channels

        for i in range(reshaped_image.shape[2]):
            ax = axes[i] if reshaped_image.shape[2] > 1 else axes
            im = ax.imshow(reshaped_image[:, :, i], cmap='hot', interpolation='nearest')

            # Increased title font size
            ax.set_title(f'{channel_names[i]} Channel', fontsize=16)  # Adjust font size as needed
            ax.axis('off')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 0.5, 1])  # Set fixed ticks
            cbar.ax.set_yticklabels(['0 (Dry)', '0.5', '1 (Moist)'])  # Set the labels after ticks are defined

        plt.tight_layout()
        plt.savefig(output_filename)  # Save the combined heatmap
        plt.close()

        logging.info(f"Saved combined heatmap: {output_filename}")  # Log the saved filename

    except Exception as e:
        logging.exception(f"Error creating combined heatmap: {e}")
        raise




# Route for rendering the HTML page
@app.route('/')
def index():
    return render_template('index.html')  # Ensure your HTML file is named index.html

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.error("No image uploaded.")
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        logging.error("No selected file.")
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Read the image file
        image_data = file.read()
        if not image_data:
            logging.error("No image uploaded.")
            return jsonify({'error': 'No image uploaded.'}), 400

        processed_image = preprocess_image(image_data)

        # Make predictions using each model
        rf_prediction = rf_model.predict(processed_image)
        cat_prediction = cat_model.predict(processed_image)
        xgb_prediction = xgb_model.predict(processed_image)

        # Create heatmap from .h5 file
        create_heatmap_from_h5()
        # create_combined_heatmap_from_h5

        # Calculate average prediction
        average_prediction = np.mean([rf_prediction, cat_prediction, xgb_prediction])



        # Format the prediction with 3 decimal places and units
        formatted_prediction = f"{average_prediction:.3f} cubic meters of water per cubic meter of soil"


        return jsonify({
            'rf_prediction': rf_prediction[0].tolist(),
            'cat_prediction': cat_prediction[0].tolist(),
            'xgb_prediction': xgb_prediction[0].tolist(),
            'average_prediction': formatted_prediction, # Return the formatted string
            'plot': '/heatmap'
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500  # Return error message with 500 status code




# Route to display the heatmap image
@app.route('/heatmap')
def heatmap():
    try:
        return send_file('heatmap.png', mimetype='image/png')
    except Exception as e:
        logging.error(f"Error sending heatmap: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
