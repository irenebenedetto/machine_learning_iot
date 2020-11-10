import io
import pyaudio
import wave
import time
import sys
import numpy as np
from os import path
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
import subprocess
from subprocess import Popen
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--num-samples',type=int, help='input file folder')
parser.add_argument('--output',help='period in seconds')

parameter = parser.parse_args()
num_sample = parameter.num_samples
output_file = parameter.output

RATE, CHANNELS, CHUNK_SIZE, L, FORMAT = 48000, 1, 1024, 1, pyaudio.paInt16
output_fts, num_mel_bins, lower_frequency, upper_frequency = 10, 40, 20, 4000
UP, DOWN = 1, 3
frame_length, frame_step = 640, 320

#Reset the monitor and set powersave
Popen(['sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"'], shell=True)
#Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)

buffer = io.BytesIO()
num_spectrogram_bins = 321
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, RATE, lower_frequency, upper_frequency)
linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix,dtype=tf.float32)

p = pyaudio.PyAudio()
time.sleep(2)
stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input=True)
for sample in range(num_sample):

    # record the audio
    # p = pyaudio.PyAudio()
    #tp = time.time()
    #Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
    #print(f"tempo popen {time.time()-tp}")
    #stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input=True)
    frames = []

    buffer.seek(0)  # pointer rewind

    # Start the counter
    t1 = time.time()
    stream.start_stream()
    #subprocess.Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)

    for i in range(int(RATE/CHUNK_SIZE)*L):
        data = stream.read(CHUNK_SIZE,False)
        frames.append(data)
        if i==0:
            subprocess.Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
        if i==8:
            #tp = time.time()
            subprocess.Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
            #print(f'{time.time()-tp}') 
    #Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
    #Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
    # stream.stop_stream()
    # stream.close()
    # p.terminate()
    stream.stop_stream()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    buffer.seek(0)
    rate, audio = wavfile.read(buffer)
    

    # undersample the signal to 16000 Hz
    audio = signal.resample_poly(audio, UP, DOWN) # the resampling is UP/DOWN of the original sampling rate
    audio = tf.cast(audio,dtype=tf.float32)

    # compute the STF
    # print("VF before stft")
    # Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'], shell=True)
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # compute mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :output_fts]
    mfccs = tf.transpose(mfccs)
    #Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
    new_file_name = output_file +'/mfccs'+str(sample) + '.bin'
    mfcc_serialized = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(new_file_name,mfcc_serialized)
    preproc_time = time.time() - t1
    print(f"{preproc_time}")
   # Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)

#Print the total time for the different VF levels
Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)
p.terminate()
stream.stop_stream()
stream.close()
