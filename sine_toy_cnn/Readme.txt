This folder contains all the necessary files to deploy a toy CNN to the Arduino Nano 33 BLE Board.

- tflite_sinewave_training.ipynb:
Python file that trains a toy neural network outputing sin(x) for any float input x.
Also transcribes the trained model into bytes (sine_model.tflite) and from the bytes into a C header file (sine_model.h).

- sine_model.tflite:
File containing the CNN in bytes format.

- sine_model.h:
File containing the CNN encoded into bytes in a format readable by C.
This file must be in the same folder as sine.ino when uploaded to the arduino board.

- sine.ino:
Arduino code responsible for the setup of the tinyml environment and the setup of the model.
When uploaded to the arduino board, this generates a sine wave based on the results of the cnn which outputs sin(x) and a given frequency.

