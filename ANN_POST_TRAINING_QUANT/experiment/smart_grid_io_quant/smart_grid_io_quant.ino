#include "model.h" //Name of the Model !!!!!!!!!!!!!!!!!

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

const int no_features = 12;

//unstable input
float inp_t[no_features] = {1.152211453, 0.6558085345,  -1.170751324,  -0.1234479822, 0.1727034534,  -0.0832047194, 0.7784549532,  -0.993445077,  -0.09288147801,  1.344446735, -0.6030629884, -0.9837311252};

//stable input
//float inp_t[no_features] = {1.347359941, -0.7658224517, -0.9103893018, -1.192112847,  0.3204514994,  0.003685512382,  0.1952717344,  -0.7540456146, 1.563269893, 0.1199067795,  -0.08084752539,  0.1495043104};


// TensorFlow Lite for Microcontroller global variables
const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;
tflite::MicroErrorReporter tflu_error;


constexpr int tensor_arena_size = 4 * 1024;//hyperparameter!!!!!
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));
float   tflu_i_scale      = 0.0f;
float   tflu_o_scale      = 0.0f;
int32_t tflu_i_zero_point = 0;
int32_t tflu_o_zero_point = 0;

inline int8_t quantize(float x, float scale, float zero_point)
{
  return (x / scale) + zero_point;
}

inline float dequantize(int8_t x, float scale, float zero_point)
{
  return ((float)x - zero_point) * scale;
}


/////////////////////////////////////////////////////////////////////
void tflu_initialization()// Model Initialization
{
  Serial.println("TFLu initialization - start");

  // Load the TFLITE model
  tflu_model = tflite::GetModel(TFLite_Models_model_tflite);//NAME OF THE MODEL BUT INSTEAD OF. USE _!!!!, path is also included in the name
  if (tflu_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("error");
    Serial.println(tflu_model->version());
    Serial.println("");
    Serial.println(TFLITE_SCHEMA_VERSION);
    Serial.println("");
    while(1);
  }
  
  tflite::AllOpsResolver tflu_ops_resolver;
  
  // Initialize the TFLu interpreter
  tflu_interpreter = new tflite::MicroInterpreter(tflu_model, tflu_ops_resolver, tensor_arena, tensor_arena_size, &tflu_error);

  // Allocate TFLu internal memory
  tflu_interpreter->AllocateTensors();

  // Get the pointers for the input and output tensors
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);




  // Get the quantization parameters (per-tensor quantization)

  const auto* i_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quantization = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  tflu_i_scale      = i_quantization->scale->data[0];
  tflu_i_zero_point = i_quantization->zero_point->data[0];
  tflu_o_scale      = o_quantization->scale->data[0];
  tflu_o_zero_point = o_quantization->zero_point->data[0];

   Serial.println("TFLu initialization - completed");
}




//////////////////////////////////////////////////////////////////////////

void setup() {
  Serial.begin(9600);
  while (!Serial);// wait for serial initialization


  tflu_initialization();
  delay(4000);
  Serial.println("Init is done");
}


////////////////////////////////////////////////////////

void loop() {
  unsigned long timeBegin = micros();
  


  // Initialize the input tensor
  for (int i = 0; i < no_features; i++) {
    tflu_i_tensor->data.int8[i] = quantize(inp_t[i], tflu_i_scale, tflu_i_zero_point);;
  }


  // Run inference
  TfLiteStatus invoke_status = tflu_interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Error invoking the TFLu interpreter");
    return;
  }
  


  //Quant output conversion
  int8_t out_int8 = tflu_o_tensor->data.int8[0];
  float out_f = dequantize(out_int8, tflu_o_scale, tflu_o_zero_point);



   if(out_f > 0.5) {
     Serial.println("Stable");
   }
   else {
     Serial.println("Unstable");
   }

  
  //execution time calculation
  unsigned long timeEnd = micros();
  unsigned long duration = timeEnd - timeBegin;
  double averageDuration = (double)duration / 1000.0;
  Serial.println(averageDuration);
  
  Serial.println();
  delay(4000);
}
