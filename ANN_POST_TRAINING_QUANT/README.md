# Smart Grid_Stability Arduino 33 BLE SENSE TFlite

The main goal of this project is to create an ANN Network and later deploy it on Arduino 33 BLE SENSE using TensorFlow Lite. The ANN can predict whether a Smart Grid is stable or unstable (Binary Classification). The code was created on Google Collab, thus making it easy to replicate. Essentially the only thing that needS to be changed is the folder path (but keeping the folder structure identical to the one on Github).

***IMPORTANT***
- A few words about the arduino storage space, which will be useful later. Arduino has 1MB of flash memory and 256KB of SRAM. Flash memory is where the arduino sketch is stored. SRAM is where the variables created by the sketch will be stored.
- TFlite model when deployed to a microcontroler doesn't require a dynamic memoryt allocation. All the memory that will be needed by the input,output,intermediate activations etc. is predefined in the sketch. The size that they will "occupy" is defined by the variable tensor arena. This memory allocation takes "space" from SRAM.

There are 3 sections:

1. An end to end walkthrough of the project
2. General details about the experiments that lead to the final results
3. Observations
4. Problems regarding the TFLite and Arduino Deployment

## 0 - Upload Models directly
- Go to the final folder
- Models are already in their respective folder so you can just go to either smart_grid_no_io_quant(no input-output quantization) or to smart_grid_io_quant(with input-output quantization) and upload the models to the arduino right away

## 1 - End to End walkthrough

- Go to the final folder
- Run smart_grid_stability_train.ipynb. This will generate the tensorflow model as well as the representative dataset which will be need for quantization.
- Run model_quant.ipynb and model_quant_io.ipynb. They will generate the tflite models and header files for the two quantization approaches. One is full integer quantization without affecting the input/output and the second is full integer quantization but quantizes both the input and the output of the model.
- If you wish to upload the models to the arduino go to either smart_grid_no_io_quant(no input-output quantization) or to smart_grid_io_quant(with input-output quantization) and open the respective arduino file using the ARDUINO IDE. 

***IMPORTANT***
- If you wish to alter the code and create new models you will have to move them from the TFLITE_models folder to their respective arduino folder. Either smart_grid_no_io_quant(no input-output quantization) or to smart_grid_io_quant(with input-output quantization). From IDE all you have to do is to click upload(top left)


## 2 - Experiments

- Go to the experiment folder
- There you will find the notebooks that contain all the model and quantization approaches that were tried. Only two quantization methods were able to run on the arduno though.

## 3 - Observations
- Model 3 from all the created models seemed to have the best accuracy and generalization power, thus we convert this one to a tflite one.
- Only the full integer quantization methods with and without input/output quantization worked
- Average inference time in both cases for one observation on arduino was 3.30 sec
- Significant model size reduction from roughly 1.1MB model size, quantization led to models of size roughly 300KB.
- When uploaded, the models used 29% of the FLASH memory and 37% of the SRAM memory. Should be noted that the size selected for tensor arena was 4*1024 Bytes/ 4KB, tensor arena is stored in SRAM. We can increase or decrease(x * 1024) the size of tensor arena to match the requirements of our model memory requirements as well as reduce the memory requirements.
- Accuracy wise the original model and full integer quantization without input/output quantization model had similar accuracy of 98%. Quantizing the input/output led to a significant accuracy drop to ~80%.

## 4 - Problems (Personal opinions included, I could be wrong in some. Feel free to message me or send a pull request if you believe so)

- The main issue was the outdated documentation and the fact that many functions/approaches are no longer supported by either arduino or tensorflow.
- Many microcontrollers models seem to not work correctly and their software support for TF seems to be outdated, the safest options would be ARDUINO 33 BLE SENSE and Rasberry PICO.
- It seems that only full integer quantization (with and wihout input/output quantization) seem to work. Otherwise an error called HYBRID MODELS not supported occurs. This seems to be due to optimization leaving part of the operations as floating point (hence hybrid) and mixing isn't supported in micro.
(https://github.com/tensorflow/tensorflow/issues/43386)
- Input/Output uint8 quantization is no longer supported.
(https://github.com/tensorflow/tflite-micro/issues/280)
- Input/Output int8 quantization had a significant accuracy drop.
(https://github.com/tensorflow/tflite-micro/issues/396)
