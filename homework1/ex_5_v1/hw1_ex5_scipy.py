import io
import pyaudio
import wave
import statistics
import time
import sys
import numpy as np
from os import path
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
from subprocess import Popen
import argparse

#Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)

parser=argparse.ArgumentParser()
parser.add_argument('--num-samples',type=int, help='input file folder')
parser.add_argument('--output',help='period in seconds')


parameter = parser.parse_args()
num_samples = parameter.num_samples 
output_file = parameter.output

avg_preproc_time = []

output_fts, num_mel_bins, lower_frequency, upper_frequency = 10, 40, 20, 4000
UP, DOWN = 1, 3
frame_length, frame_step = 640, 320

RATE, CHANNELS, CHUNK_SIZE, L, FORMAT = 48000, 1, 1024, 1, pyaudio.paInt16

for sample in range(num_samples):
    

    # record the audio

    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input=True)
    
    frames = []

    t1 = time.time()
    # save the wave file into a buffer
    buffer = io.BytesIO()

    for i in range(int(RATE/CHUNK_SIZE)*L):
        buffer.write(stream.read(CHUNK_SIZE))

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    buffer.seek(0)  # pointer rewind
    
    #rate, audio = wavfile.read(buffer)
    # speed up the processor frequency
    
    # undersample the signal to 16000 Hz
    audio = signal.resample_poly(np.frombuffer(buffer.getvalue()), UP, DOWN) # the resampling is UP/DOWN of the original sampling rate
    print(f"Resampling time: {time.time()-t1}")   
    
    # compute the STFT 
    _, _, spectrogram = signal.spectrogram(audio, fs=16000, nfft=640, nperseg=640, noverlap=320, mode= 'complex')
    spectrogram= tf.transpose(tf.abs(spectrogram))
    print(f"STFT+spectrogram time: {time.time()-t1}")    
    
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, RATE, lower_frequency, upper_frequency)
    linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix,dtype=tf.float64)
    
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    
    # compute mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :output_fts]
    
    # store into a bin file
    new_file_name = output_file + str(sample) + '.bin'
    mfcc_serialized = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(new_file_name,mfcc_serialized)
    print(f"MFCCS time: {time.time() -t1}")
    avg_preproc_time.append(time.time() - t1)
    # slow down the processor frequency
    #Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'],shell=True)
    
print(f"Avg preprocessing time: {statistics.mean(avg_preproc_time)} +/- {statistics.stdev(avg_preproc_time)}")
