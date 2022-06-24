This folder contains all necessary files to deploy a diabetes detection algorithm on arduino.
Current methods available:
- Quantization
- Weight clustering


Results so far:
- Clustering seems to cause size reduction of about a factor of 6.
- Clustering + quantization seems to cause size reduction of about a factor of 8.
- Model accuracy seems to be largely unimpacted by clustering and quantization.



Files contained:

	- diabetes_test.ipnyb:
Python file that imports data from the diabetes_dataset.csv file, and generates a keras model for detecting diabetes in a person based on the features given in the dataset.
This code creates a keras model, a quantized tflite model, and a tflite model which has gone through weight clustering and quantization.


	- diabetes_dataset.scv:
A csv file containing relevant data for the detection of diabetes in patients.

Input Variables (X):
    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    BloodPressure: Diastolic blood pressure (mm Hg)
    SkinThickness: Triceps skin fold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    DiabetesPedigreeFunction: Diabetes pedigree function
    Age: Age (years)

Output Variables (Y):
    Outcome: a 0 or 1 indicating if the patient has diabetes or not


	- keras_model folder:
Contains the saved keras model before any qauntization or weight clustering.


	- clustered_quantized.tflite:
Contains the tflite model after both weight clustering and quantization.



##TODO:
- Add .ino file for deployment on Arduino
- Add libraries versions required
- Clean up ipynb file
- Add pruning algorithm