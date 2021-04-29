# Homework 1

## Background 1: Sensor Fusion

Many prediction tasks can benefit from exploiting data originating from different sources. We refer to this kind of applications with *sensor fusion*. Some popular examples are: 
1. scene classifications, which relies on image, audio, and temperature to classify the environment (indoor or outdoor); 
2. position and velocity estimation, which combines several inertial sensors (accelerometer, gyroscope, magnetometer, altimeter); 
3. activity recognition, which can complement measurements from inertial sensors with electrocardiography and electromyography data. 

Inspired by humans that use the five basic senses to sense and process information, sensor fusion relies on the hypothesis that the analysis of multiple sources of data improves the prediction accuracy. In sensor fusion applications, the training pipeline is therefore fed with structured data. An efficient
implementation requires to organize and prepare the data in order to create datasets compliant with the specifications of standard training frameworks.

## HowTo 2: TFRecord Datasets

Data movement represents one of the major bottlenecks in neural network training. Indeed, standard training pipelines encompasses the transfer of a massive amount of data from the main storage to the computational units, e.g. from the hard drive to the GPU memory. Fetching data from large storages is slow and energy costly. To mitigate this issue, raw data should be packed in serialized sequences of binary records. In this way, the data will reside in consecutive locations of memory, thus accelerating parallel fetching operations.


The TensorFlow Data API introduced the *TFRecord* format (see `tf.train.TFRecord`) to enable users to easily serialize structured data. Specifically, the user needs to specify:

- the structure of each record through the definition of a key-value based dictionary, called *Example* (see *tf.train.Example*);
- the value for each key of the record, i.e. the *Feature* (see *tf.train.Feature*);
- the data-type for each value (see *tf.train.BytesList*, *tf.train.FloatList*, *tf.train.Int64List*)



## Background 3: Power management with Dynamic Voltage Frequency Scaling

Dynamic Voltage Frequency Scaling (DVFS) is today a standard power management strategy for System-on-Chips (SoCs). It relies on the observation that the power consumption of a digital circuit has a quadratic dependance from voltage (*V* ) and a linear dependance from clock frequency (*f* ).

Voltage Frequency (VF) scaling therefore enables a cubic reduction of power.

To limit the power consumption, the SoC should run at maximum performance (VFmax) only for short periods, e.g. to meet tight timing constraints in critical and intensive tasks. Otherwise, the SoC can operate into a low-power state (VFmin) and save power without impact on the user experience.


## HowTo 4: DVFS on the Raspberry Pi

On Unix-based operating systems (e.g. Raspian), the DVFS is managed by an operating system routine called governor. The standard governor can select two main policies: *performance* and *powersave*. In *performance* mode, the VF level is set to the maximum value VFmax=0.86V @ 1.5GHz, whereas in *powersave* mode, it is set to the minimum value VFmin=0.81V @ 600MHz.

To manually change the operating mode, modify the content of the *scaling\_governor* system file:


```bash
#Set 1.5GHz
sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling\_governor"

# Set 600MHz
sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling\_governor"
```
To check the current VF, read the *file* system file, that stores the operating clock frequency in Hertz:
```bash
cat /sys/devices/system/cpu/cpufreq/policy0/scaling\_cur\_freq
```
To monitor the operating time at different VF levels, the operating system provides performance monitors, that work as follows:
```bash
# Reset the monitors

sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"

# Read the monitors

cat /sys/devices/system/cpu/cpufreq/policy0/stats/time\_in\_state
```
Note that there exists intermediate VF levels and other governor policies, which will be not considered in this homework.

## Exercise 4: Data Preparation: Sensor Fusion Dataset with TFRecord (3pt)

Write a Python script on your notebook to build a *TFRecord* Dataset containing temperature, humidity, and one-second audio samples (Int16 resolution at 48KHz), collected at different datetimes and stored as raw data in a single folder as shown in following example:
```bash
./raw_data

samples.csv

audio1.wav
audio2.wav
audio3.wav
audio4.wav
```

Content of *samples.csv*:

```bash
18/10/2020,09:45:34,21,65,audio1.wav
18/10/2020,09:46:40,21,65,audio2.wav
18/10/2020,09:47:45,21,65,audio3.wav
18/10/2020,09:48:51,21,65,audio4.wav
```
**N.B.:** The *raw\_data* folder is an input of the script. The script should contain only the code to build

the *TFRecord* Dataset.

Each *Example* in the *TFRecord* must have four fields:
- datetime (stored as POSIX timestamp);
- temperature;
- humidity;
- audio.

Select the proper datatype depending on the field in order to minimize the storage requirements while keeping the original quality of the data. Motivate your choices. No assumptions must be done on the number of records contained in the dataset.

SYNOPSIS:
```bash
python HW1\_ex4\_GroupN.py --input <INPUT DIR> --output <OUTPUT FILE>
```
EXAMPLE:
```bash
python HW1\_ex4\_Group1.py --input ./raw\_data --output ./fusion.tfrecord
```
## Exercise 5: Low-power Data Collection and Pre-processing (3pt)

Write a Python script that iteratively samples 1-second audio signals, process the MFCCs, and store the output on disk. The script takes as input the number of samples to collect and the output path where to store the MFCCs.

Set the resampling, STFT, and MFCCs parameters to the same values as Lab 2 – Ex6.

The pre-processing latency must be <80ms (hard constraint). Identify which steps of the acquisition and pre-processing loop can be performed @VFmin such that the time @VFmin is maximized and the pre-processing latency is met.

The script should print the latency of each iteration of the loop and the total time spent at the different VF levels. Comment the results.

**Suggestion #1:** Avoid writing intermediate data on disk to reduce the processing time (check the *BytesIO* class from the *io* package).

**Suggestion #2:** Use the *Popen* class from the *subprocess* package to run DVFS commands in non-blocking mode.


SYNOPSIS:
```bash
python HW1\_ex5\_GroupN.py --num-samples <NUM SAMPLES> --output <OUTPUT>
```
EXAMPLE:
```bash
python HW1\_ex5\_Group1.py --num-samples 5 --output ./out
```
Output:

```bash
1.054
1.069
1.068
1.069
1.068

600000 465
750000 0
1000000 0
1500000 64
```
Content of the *out* folder:
```bash
./out

mfccs0.bin
mfccs1.bin
mfccs2.bin
mfccs3.bin
mfccs4.bin
```