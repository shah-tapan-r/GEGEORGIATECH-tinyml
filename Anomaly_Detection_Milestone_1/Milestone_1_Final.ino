/**
 * Use TensorFlow Lite model on real accelerometer data to detect anomalies
 * 
 * NOTE: You will need to install the TensorFlow Lite library:
 * https://www.tensorflow.org/lite/microcontrollers
 * 
 * Author: Akshay Sridharan
 * 
 * 
 */

// Library includes
#include <Adafruit_LIS3DH.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>


// Local includes
#include "c_model.h"

// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//Import Arcada libraries for operating device speaker and screen
#include <Adafruit_Arcada.h>
#include <Adafruit_Arcada_Def.h>

// Used for software SPI
#define LIS3DH_CLK 13
#define LIS3DH_MISO 12
#define LIS3DH_MOSI 11
// Used for hardware & software SPI
#define LIS3DH_CS 10

Adafruit_Arcada arcada; // Set up the Arcada code

// We need our utils functions for calculating MAE
extern "C" {
#include "utils.h"
};

// Set to 1 to output debug info to Serial, 0 otherwise
#define DEBUG 1

// Pins
constexpr int BUZZER_PIN = A0;

// SOUND
#define sound_pin A0     // Direct sound 
uint8_t sound_on = 1;     // start with sound on

// Settings
constexpr int NUM_AXES = 3;           // Number of axes on accelerometer
constexpr int MAX_MEASUREMENTS = 16; // Number of samples to keep in each axis// Globals
constexpr float THRESHOLD = 0.4;    // Any MAE over this is an anomaly
constexpr int WAIT_TIME = 0;       // ms between sample sets
constexpr int SAMPLE_RATE = 25;      // How fast to collect measurements (Hz)

//Initializing Accelerometer
Adafruit_LIS3DH lis = Adafruit_LIS3DH(&Wire);

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 5 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace
 
/*******************************************************************************
 * Main
 */
 
void setup() {

  // Initialize Serial port for debugging
#if DEBUG
  Serial.begin(115200);
  while (!Serial);
#endif

  if (!arcada.arcadaBegin()) {
    Serial.print("Failed to begin");
    while (1);
  }

  arcada.enableSpeaker(1);   // Enable the speaker and play opening tone
  pinMode(sound_pin, OUTPUT);
  beep(sound_pin,  880, 100);

  // Initialize accelerometer
  if (!lis.begin()) {
#if DEBUG
    Serial.println("Failed to initialize LIS3DH");
#endif
    while (1) yield();
  }

  // Set and print the accelerometer range
  // Synthetic data for example was collected at a 4G range
  lis.setRange(LIS3DH_RANGE_4_G);   // 2, 4, 8 or 16 G!
  Serial.print("Range = ");
  Serial.print(2 << lis.getRange()); 

  // Configure buzzer pin
  pinMode(BUZZER_PIN, OUTPUT);

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(c_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
 
 }

void loop() {

  float sample[MAX_MEASUREMENTS][NUM_AXES];
  float measurements[MAX_MEASUREMENTS];
  float y_val[NUM_AXES];
  float mae;
  TfLiteStatus invoke_status;
  
  // Timestamps for collecting samples
  static unsigned long timestamp = millis();
  static unsigned long prev_timestamp = timestamp;

  // Take a given time worth of measurements
  int i = 0;
  while (i < MAX_MEASUREMENTS) {
    if (millis() >= timestamp + (1000 / SAMPLE_RATE)) { 
  
      // Update timestamps to maintain sample rate
      prev_timestamp = timestamp;
      timestamp = millis();

      // Take sample measurement
      sensors_event_t event;
      lis.getEvent(&event);

      // Add normalized readings to array
      // Units of milli-g's will be used
      sample[i][0] = -1*event.acceleration.z / 9.8;
      sample[i][1] = event.acceleration.x / 9.8;
      sample[i][2] = event.acceleration.y / 9.8;

      Serial.print(sample[i][0], 2);
      Serial.print(", "); Serial.print(sample[i][1], 2);
      Serial.print(", "); Serial.println(sample[i][2], 2);
      // Update sample counter
      i++;
    }
  }
 

  // Print out samples
#if DEBUG
  Serial.print("Samples: ");
  for (int i = 0; i < MAX_MEASUREMENTS; i++) {
    for (int axis = 0; axis < NUM_AXES; axis++) {
      Serial.print(sample[i][axis], 7);
      Serial.print(" ");
      }
      Serial.println();
  }
  
#endif

for (int i = 0; i < MAX_MEASUREMENTS; i++) {
  // Copy sample values to input buffer/tensor
  for (int axis = 0; axis < NUM_AXES; axis++) {
    model_input->data.f[axis] = sample[i][axis];
  }

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  for (int axis = 0; axis < NUM_AXES; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  // Calculate MAE between given and predicted values
  mae = calc_mae(sample[i], y_val, NUM_AXES);

  // Print out result
#if DEBUG
  Serial.print("Sample Values: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.println(sample[i][axis], 7);
  }
  Serial.print("Inference result: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.println(y_val[axis], 7);
  }
  Serial.println(); 
  Serial.print("MAE: "); 
  Serial.println(mae, 7);
#endif

  // Compare to threshold
  if (mae > THRESHOLD) {
    pinMode(sound_pin, OUTPUT);
    beep(sound_pin,  880, 100);

#if DEBUG
    Serial.println("DANGER!!!");
#endif
  } else {
  }
#if DEBUG
  Serial.println();
#endif
}
  delay(WAIT_TIME);

}

// A sound-producing function
void beep(unsigned char speakerPin, int frequencyInHertz, long timeInMilliseconds) {
  // http://web.media.mit.edu/~leah/LilyPad/07_sound_code.html
  int  x;
  if( sound_on ) {     // Only play if the sound_on flag is 1
    long delayAmount = (long)(1000000 / frequencyInHertz);
    long loopTime = (long)((timeInMilliseconds * 1000) / (delayAmount * 2));
    for (x = 0; x < loopTime; x++) {
      digitalWrite(speakerPin, HIGH);
      delayMicroseconds(delayAmount);
      digitalWrite(speakerPin, LOW);
      delayMicroseconds(delayAmount);
    }
  }
}
