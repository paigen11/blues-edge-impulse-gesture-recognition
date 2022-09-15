#include <Adafruit_LIS3DH.h>
#include <ei-gesture-recognition_inferencing.h>

static bool debug_nn = false;

Adafruit_LIS3DH lis = Adafruit_LIS3DH();

void setup(void)
{
  Serial.begin(115200);
  lis.begin(0x18);

  if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 3)
  {
    ei_printf("ERR: EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME should be equal to 3 (the 3 sensor axes)\n");
    return;
  }
}

void loop()
{
  // lis.read(); // get x,y,z data at once
  // Serial.print(lis.x);
  // Serial.print("\t");
  // Serial.print(lis.y);
  // Serial.print("\t");
  // Serial.print(lis.z);
  // Serial.println();

  ei_printf("\nStarting inferencing in 2 seconds...\n");
  delay(2000);
  ei_printf("Sampling...\n");

  // Allocate a buffer here for the values we'll read from the IMU
  float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = {0};

  for (size_t ix = 0; ix < EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE; ix += 3)
  {
    // Determine the next tick (and then sleep later)
    uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

    lis.read();
    buffer[ix] = lis.x;
    buffer[ix + 1] = lis.y;
    buffer[ix + 2] = lis.z;

    delayMicroseconds(next_tick - micros());
  }

  // Turn the raw buffer in a signal which we can the classify
  signal_t signal;
  int err = numpy::signal_from_buffer(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
  if (err != 0)
  {
    ei_printf("Failed to create signal from buffer (%d)\n", err);
    return;
  }

  // Run the classifier
  ei_impulse_result_t result = {0};

  err = run_classifier(&signal, &result, debug_nn);
  if (err != EI_IMPULSE_OK)
  {
    ei_printf("ERR: Failed to run classifier (%d)\n", err);
    return;
  }

  ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)\n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
  uint8_t predictionLabel = 0;
  for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
  {
    Serial.print("    ");
    Serial.print(result.classification[ix].label);
    Serial.print(": ");
    Serial.println(result.classification[ix].value);

    if (result.classification[ix].value > result.classification[predictionLabel].value)
      predictionLabel = ix;
  }

  // print the predictions
  String label = result.classification[predictionLabel].label;

  Serial.print("\nPrediction: ");
  Serial.println(label);

#if EI_CLASSIFIER_HAS_ANOMALY == 1
  ei_printf("    anomaly score: %.3f\n", result.anomaly);
#endif
}