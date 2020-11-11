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
from subprocess import Popen
import argparse
from threading import Thread

def performance():
    Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)

def powersave():
    Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)

parser=argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, help='input file folder')
parser.add_argument('--output', help='period in seconds')

parameter = parser.parse_args()
num_sample = parameter.num_samples
output_file = parameter.output

RATE, CHANNELS, CHUNK_SIZE, L, FORMAT = 48000, 1, 1024, 1, pyaudio.paInt16
output_fts, num_mel_bins, lower_frequency, upper_frequency = 10, 40, 20, 4000
UP, DOWN = 1, 3
frame_length, frame_step = 640, 320

buffer = io.BytesIO()
num_spectrogram_bins = 321
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, RATE, lower_frequency, upper_frequency)
linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix, dtype=tf.float32)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)
#Reset the monitor and set powersave
Popen(['sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"'], shell=True)

for sample in range(num_sample):

    t2 = time.time()
    stream.start_stream()
    buffer.seek(0)  # pointer rewind

    # it does 46 cycles
    # each cycle takes 21.74 milliseconds
    # the Popen instantiation takes 50 milliseconds
    # th Popen instantiation + scaling_governor take 150 millisecond --> 50 milliseconds are "sunk"
    # anticipate 100/21.74 = 4 for loops --> scaling_governor at iteration i = 45-9 = 41

    for i in range(int(RATE/CHUNK_SIZE*L)):
        buffer.write(stream.read(CHUNK_SIZE, exception_on_overflow=False))
        if i == 38:
            t = Thread(target=performance)
            t.start()

    t1 = time.time()
    stream.stop_stream()
    buffer.seek(0)

    # undersample the signal to 16000 Hz
    audio = signal.resample_poly(np.frombuffer(buffer.getvalue(), dtype=np.int16), UP, DOWN) # the resampling is UP/DOWN of the original sampling rate
    audio = tf.cast(audio, dtype=tf.float32)

    # compute the STF
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape((48, 40))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # compute mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :output_fts]

    new_file_name = output_file + '/mfccs' + str(sample) + '.bin'
    mfcc_serialized = tf.io.serialize_tensor(mfccs)

    tf.io.write_file(new_file_name, mfcc_serialized)
    t = Thread(target=powersave)
    t.start()
    print(f"Complete time: {time.time() - t2}")
    print(f'Preprocessing only: {time.time() - t1}')



stream.close()
p.terminate()
#Print the total time for the different VF levels
Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)
