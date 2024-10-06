**Soil Moisture Prediction System**
This repository provides a comprehensive pipeline for predicting soil moisture using various machine learning models. The system is divided into three main components:

Data Preparation and Modeling (main.py)
Web Application (app.py)
Frontend Interface (index.html)

**1. Data Preparation and Modeling (main.py)**
The main.py script handles the preprocessing of soil moisture data and the training of machine learning models. The process involves:

Loading Data: The script loads soil moisture data from an HDF5 file.
Handling Missing Data: It performs data cleaning by addressing any missing data points.
Data Segmentation: The script segments the data to create meaningful inputs for machine learning models.
Modeling: Several regression models are trained, including:
Random Forest
CatBoost
XGBoost
Evaluation: Model performance is evaluated using the following metrics:
R-squared
Root Mean Squared Error (RMSE)
Model Saving: The trained models are saved to disk for future predictions.

**2. Web Application (app.py)**
The app.py script powers the web-based interface using the Flask framework. Key functionalities include:

Loading Pre-trained Models: The script loads the models previously trained and saved by main.py.
Image Upload Handling: Users can upload images of soil, which are processed to predict soil moisture.
Soil Moisture Prediction: Based on the input image, the script predicts the soil moisture level using the loaded models.
Heatmap Generation: The script generates and saves heatmaps visualizing moisture data, which can be viewed in the web interface.

**3. Frontend Interface (index.html)**
The index.html file provides a simple and user-friendly interface for interacting with the web application. Key features include:

Image Upload: Users can upload images of soil through a file upload interface.
Dynamic Display of Results: Predictions of soil moisture and any generated heatmaps are displayed dynamically.
Visualization: Generated heatmaps visualizing soil moisture predictions can be viewed directly on the page.
