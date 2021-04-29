# Homework 3
## Exercise 1: Big/Little Inference
Develop a Big/Little model for Keyword Spotting composed by a Big neural network running on your notebook and a Little neural network running on your Raspberry Pi. Both models must be deployed in ```tflite``` format.
The overall system must be composed by:

- A web service (named ```big_service.py```) that receives a raw audio signal (sampling frequency is 16 kHz) in SenML+JSON format, runs inference using the Big model, and returns the output label (in JSON format).
SYNOPSIS:
```bash
python big_service.py
```

- A client application (```little_client.py```) that sequentially reads an audio signal from the Speech Command test-set, runs inference using the Little model and, if needed, invokes the web service to get the prediction from the Big model. Define and implement the “success checker” policy that decides whether to accept the prediction of the Little model or invoke the web service.

The client must print on the screen the accuracy and the communication cost measured on the Speech Command test-set. The communication cost is the sum among the sizes of the SenML+JSON strings sent to the web-service.
 
SYNOPSIS:
```bash
python little_client.py
```
EXAMPLE OUTPUT:
```bash
Accuracy: 93.125%
Communication Cost: 4.405 MB
```
Identify and adopt the most suitable protocol (between REST and MQTT) to implement the communication among the different applications. Motivate your choice.
- Write a Python script to train the Big and Little models. For the training, use the signals listed in train_files.txt and the labels order in labels.txt from the Portale.
SYNOPSIS:
```bash
python train.py --version <big|little>
```
The models must be compliant with the following constraints (measured on the test-set):
   - Big/Little Accuracy > 93%
   - Little TFLite Size < 20 kB
   - Little Total Inference Time < 40 ms
   - Communication cost < 4.5 MB

The final submission should contain the following files:
    1. The ```big_service.py``` script.
    2. The ```little_client.py``` script.
    3. The ```train.py``` script.
    4. The Big/Little models in tflite format (```big.tflite``` and ```little.tflite```). If you have compressed
the TFLite files with zlib, append ```.zlib``` to the filenames.
    5. Any other .py file needed for the correct execution of the previous commands (if any).

Organize the files in two different folders, named notebook and rpi respectively, depending on the target device where the commands should run. In the report, briefly describe the Big/Little models and the “success checker” policy adopted. 

## Exercise 2: Cooperative Inference
Develop a client application that receives a pre-processed audio signal, classify its content, and send the output of the last neural network layer (in JSON format). Run different $N$ ($N$ is an optimization parameter, $N > 1$) instances of the same application, each of them running a different prediction model (in tflite format). The application instances could potentially run on different devices (suggestion: launch them on different shells on your notebook). To run the application, develop a Python command with the following specifications:
 
SYNOPSIS:
```bash
python inference_client.py --model <model_path>
```
where model_path is the path to the tflite model.
On your Raspberry Pi, develop a client application that sequentially read an audio signal from the Speech Command test-set (the signals are listed in test_files.txt from the Portale), apply some pre-processing, send the pre-processed signal (in SenML+JSON format) to the inference clients, and retrieve the neural networks outputs. Define and implement a cooperative inference policy to compute the final prediction combining the outcomes of the different inference clients. To run the application, develop a Python command with the following specifications:
SYNOPSIS:
```bash
python cooperative_client.py
```
The command should print on the screen the classification accuracy measured on the Speech Command test-set.
EXAMPLE OUTPUT:
```
Accuracy: 94.375%
```
Identify and adopt the most suitable protocol (between REST and MQTT) to implement the communication among the different clients. Motivate your choice.
The application must reach an Accuracy > 94% (measured on the test-set).
The final submission should contain the following files:
1. The ```inference_client.py``` script.
2. The ```cooperative_client.py``` script.
3. The N tflite models (named as ```1.tflite```, ```2.tflite```, ..., ```N.tflite```).
4. The training script used to generate the tflite models. The training command should meet the following specifications:
SYNOPSIS:
```bash
python train.py --version <ID>
```
where $<ID>$ is an integer number ranging from 1 to $N$.
5. Any other .py file needed for the correct execution of the previous commands (if any).

Organize the files in two different folders, named notebook and rpi respectively, depending on the target device where the commands should run.
In the report, briefly describe the models used for inference and the cooperative policy adopted.