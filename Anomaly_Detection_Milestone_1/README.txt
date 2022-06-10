This folder contains all of the necessary files to deploy an accelerometer anomaly detection workflow to the Adafruit EdgeBadge.

	⁃	Milestone_1_Final.ipynb
Python file that imports necessary data and trains an autoencoder neural network. It then identifies an appropriate anomaly detection threshold and exports the trained model as a C header file. Instructions for running file are included in markdown cells. Should not run data collection cell unless new data must be created. Instead may run subsequent cells to import previously generated data from df.csv. Should not be included in folder uploaded to Adafruit.

	⁃	df.csv
Training and test data obtained from Adafruit magic_wand example to be used in Milestone_1_Final.ipynb. Must be in the same working directory as Milestone_1_Final.ipynb.


	⁃	c_model.h
Header file containing the trained autoencoder. Must be included in folder uploaded to Adafruit.

	⁃	Milestone_1_Final.ino
Arduino script responsible for the setup of the tinyml environment and the setup of the model.
When uploaded to the Adafruit board, this conducts inference on live accelerometer data and activates the piezo buzzer speaker when an anomaly is detected. Must be included in folder uploaded to Adafruit.

	⁃	utils.h/utils.c
C and Header file containing useful functions for use in anomaly detection pipeline. Of primary use in this workflow is the calc_mae function that quickly calculates the mean absolute error between two arrays. Must be included in folder uploaded to Adafruit.