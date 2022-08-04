This folder contains further folders containing all of the necessary files to deploy an accelerometer anomaly detection workflow to the Adafruit EdgeBadge.

	⁃	Milestone_1_Final and Milestone_1_Final_Voting
Contains all necessary files for a live demo of the implementation. Load all files in either of these folders into the Arduino device in a folder of the same name. These folders contain the respective .ino, and the necessary C Array file, .h file, and .c file for computational operations.

The .ino files contained are the Arduino scripts responsible for the setup of the tinyml environment and the setup of the model.
When uploaded to the Adafruit board, this conducts inference on live accelerometer data and activates the piezo buzzer speaker when an anomaly is detected. Must be included in folder uploaded to Adafruit.

The utils.h and utils.c files C and Header file contain useful functions for use in anomaly detection pipeline. Of primary use in this workflow is the calc_mae function that quickly calculates the mean absolute error between two arrays. Must be included in folder uploaded to Adafruit.

	⁃	df.csv
Training and test data obtained from Adafruit magic_wand example to be used in Milestone_1_Final.ipynb. Must be in the same working directory as Milestone_1_Final.ipynb.

	⁃	Model_3_Files and Model_4_Files
Contains header files and .tflite files containing the trained autoencoder. Also contains Python files that import necessary data and train an autoencoder neural network. It then identifies an appropriate anomaly detection threshold and exports the trained model as a C header file. Instructions for running file are included in markdown cells. Should not run data collection cell unless new data must be created. Instead may run subsequent cells to import previously generated data from df.csv. Should not be included in folder uploaded to Adafruit.

Important note on necessary libraries:

- Arduino:
In order to run on Adafruit the following libraries must be installed in the Arduino IDE:
	Board Packages:
		1. Adafruit SAMD Boards (32-bits ARM Cortex-M0+ v.1.8.13
		
	Libraries:
		1. Arduino_TensorFlowLite v.2.1.0-ALPHA
		2. Adafruit TensorFlow Lite v.1.2.3
		3. Adafruit LIS3DH v.1.2.3
		4. Adafruit Arcada Library v.2.5.3
		

