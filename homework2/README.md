# Homework 2
## Exercise 1: Multi-Step Temperature and Humidity Forecasting 
Write a Python script to train multi-output models for temperature and humidity forecasting to infer multi-step predictions, i.e. a sequence of future values. Set the number of output steps to 6. Use the Jena Climate Dataset with a 70%/20%/10% train/validation/test split (same as Lab 3).

- Implement a data-preparation pipeline compliant with multi-step predictions. Specifically, the labels shape should be ```[#Batches, #Output Steps, #Features]```, e.g. [32, 6, 2]. Use the WindowGenerator class of Lab3 as starting point.

- Implement multi-output/multi-step models, i.e. with an output shape equal to ```[#Batches, #Steps, #Features]```. Use the models developed in Lab3 as starting point.
  
- Implement a Keras metric that computes the mean absolute error of temperature and humidity on multi-step predictions (the error shape is ```[#Features]```). Use the error metric developed in Lab3 as starting point.

- Train two different model versions, each one meeting the following constraints, respectively: 
   - Version a): T MAE < 0.5°C and Rh MAE < 1.8% and TFLite Size < 2 kB
   - Version b): T MAE < 0.6 °C and Rh MAE < 1.9% and TFLite Size < 1.7 kB
N.B: The models must be trained on the training set only and evaluated on the test set.

- Submit the TFLite models (named GroupN_th_a.tflite and GroupN_th_b.tflite), together with one single Python script to train and optimize them. If you have compressed the TFLite file with zlib, append .zlib to the filename.

The script should take as input argument the model version:
```bash
python HW2_ex1_GroupN.py -–version <VERSION>
```
where $N$ is the group ID and ```<VERSION>``` is “a” or “b”, and return as output the TFLite file.

- In the report, explain and motivate the methodology adopted to meet the constraints (discuss
on model architecture, optimizations, hyper-parameters, etc.).

## Exercise 2: Keyword Spotting 
Write a Python script to train models for keyword spotting on the original mini speech command dataset. Use the train/validation/test splits provided in the Portale.
Train three different model versions, each one meeting the following constraints, respectively: 
- Version a): Accuracy > 90% and TFlite Size < 25 kB
- Version b): Accuracy > 90% and TFlite Size < 35 kB and Inference Latency < 1.5 ms 
- Version c): Accuracy > 90% and TFlite Size < 45 kB and Total Latency < 40 ms

To measure Latency, run the script kws_latency.py provided in the Portale.
Submit the TFLite models (named GroupN_kws_a.tflite, GroupN_kws_b.tflite, GroupN_kws_c.tflite), together with one single Python script to train and optimize them. If you have compressed the TFLite file with zlib, append .zlib to the filename.
The script should take as input argument the model version:

```bash
     python HW2_ex2_GroupN.py -–version <VERSION>
```
where $N$ is the group ID and ```<VERSION>``` is “a”, “b”, or “c”, and return the TFLite file.
In the report, explain and motivate the methodology adopted to meet the constraints (discuss on pre-processing, model architecture, optimizations, hyper-parameters, etc.).