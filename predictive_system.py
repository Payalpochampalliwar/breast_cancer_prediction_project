# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('C:/Users/sneha/Downloads/Disease_Project/trained_model.sav', 'rb'))

input_data = (18.94,21.31,123.6,1130,0.09009,0.1029,0.108,0.07951,0.1582,0.05461,0.7888,0.7975,5.486,96.05,0.004444,0.01652,0.02269,0.0137,0.01386,0.001698,24.86,26.58,165.9,1866,0.1193,0.2336,0.2687,0.1789,0.2551,0.06589)

# Preparing the input data
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Making a prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
   print('The Breast Cancer is Malignant')
else:
   print('The Breast Cancer is Benign')

