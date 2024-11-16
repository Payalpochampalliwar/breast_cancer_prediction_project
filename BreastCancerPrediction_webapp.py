# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:41:12 2024

@author: sneha
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/sneha/Downloads/Disease_Project/trained_model.sav', 'rb')) 

#creating function for prediction

def breastcancer_prediction(input_data):
    
    # changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    #reshape array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Making a prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
       return 'The Breast Cancer is Malignant'
    else:
       return 'The Breast Cancer is Benign'
   
def main():
    
    #giving a title for web page interface
    st.title('Breast Cancer Prediction Web App')
    
    #getting input data from the user
    
    mean_radius = st.text_input('Mean Radius Value')
    mean_texture = st.text_input('Mean Texture Value')
    mean_perimeter = st.text_input('Prerimeter mean Value')
    mean_area = st.text_input('Mean Area Value')
    mean_smoothness = st.text_input('Mean Smoothness Value')
    mean_compactness = st.text_input('Mean Compactness Value')
    mean_concavity = st.text_input('Mean Concavity Value')
    mean_concave_points = st.text_input('Concave points Value')
    mean_symmetry = st.text_input('Mean Symmetry Value')
    mean_fractal_dimension = st.text_input('Mean Fractal Value')
    radius_se = st.text_input(' Radius Value')
    texture_se = st.text_input(' Texture Value')
    perimeter_se = st.text_input(' Perimeter Value')
    area_se = st.text_input(' Area Value')
    smoothness_se = st.text_input(' Smoothness Value')
    compactness_se = st.text_input(' Compactness Value')
    concavity_se = st.text_input(' Concavity Value')
    concave_points_se = st.text_input(' Concave Points Value')
    symmetry_se = st.text_input(' Symmetry Value')
    fractal_dimension_se = st.text_input(' Fractal Dimension Value')
    worst_radius = st.text_input('Worst Radius Value')
    worst_texture = st.text_input('Worst Texture Value')
    worst_perimeter = st.text_input('Worst Perimeter Value')
    worst_area = st.text_input('Worst area Value')
    worst_smoothness = st.text_input('Worst Smoothness Value')
    worst_compactness  = st.text_input('Worst Compactness Value')
    worst_concavity = st.text_input('Worst Concavity Value')
    worst_concave_points = st.text_input('Concave Points Value')
    worst_symmetry = st.text_input('Worst symmetry Value')
    worst_fractal_dimension = st.text_input('Worst Fractal Dimension Value')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction 
    
    if st.button('Breast Cancer Test Result'):
        try:
           # Convert inputs to float and make a prediction
           diagnosis = breastcancer_prediction([
               float(mean_radius), float(mean_texture), float(mean_perimeter), float(mean_area), float(mean_smoothness),
               float(mean_compactness), float(mean_concavity), float(mean_concave_points), float(mean_symmetry),
               float(mean_fractal_dimension), float(radius_se), float(texture_se), float(perimeter_se), float(area_se),
               float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se),
               float(symmetry_se), float(fractal_dimension_se), float(worst_radius), float(worst_texture),
               float(worst_perimeter), float(worst_area), float(worst_smoothness), float(worst_compactness),
               float(worst_concavity), float(worst_concave_points), float(worst_symmetry), float(worst_fractal_dimension)
           ])
        except ValueError:
           st.error("Please enter valid numeric values for all fields.")
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()