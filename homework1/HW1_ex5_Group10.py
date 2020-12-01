import io
import pyaudio
import wave
import time
import sys
import numpy as np
import os
import sys
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
from subprocess import Popen
import argparse

os.close(sys.stderr.fileno()) 

parser=argparse.ArgumentParser()
parser.add_argument('--num-samples', type=int, help='input file folder')
parser.add_argument('--output', help='period in seconds')

parameter = parser.parse_args()
num_sample = parameter.num_samples
output_file = parameter.output

RATE, CHANNELS, CHUNK_SIZE, L, FORMAT = 48000, 1, 2400, 1, pyaudio.paInt16
output_fts, num_mel_bins, lower_frequency, upper_frequency = 10, 40, 20, 4000
UP, DOWN = 1, 3
frame_length, frame_step = 640, 320

n_cycles = int(RATE/CHUNK_SIZE)*L
n = int(n_cycles - (n_cycles * 150)/1000) -1

num_spectrogram_bins = 321
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, RATE, lower_frequency, upper_frequency)
linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix, dtype=tf.float32)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
stream.stop_stream()
Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True).wait()
#Reset the monitor and set powersave
Popen(['sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"'], shell=True).wait() 

for sample in range(num_sample):

    t_start = time.time()
    stream.start_stream()
    
    
    # it does 20 cycles
    # each cycle takes 50 milliseconds
    # the Popen instantiation takes 50 milliseconds
    # th Popen instantiation + scaling_governor take 150 millisecond --> 50 milliseconds are "sunk"
    # anticipate 100/21.74 = 3 for loops --> scaling_governor at iteration i = (20-3) -1 = 16
    with io.BytesIO() as buffer:
        if sample !=0:
            # avoid the instace at the first time
            Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)
        for i in range(n_cycles):
            data = stream.read(num_frames=CHUNK_SIZE, exception_on_overflow=False)
            buffer.write(data)
            if i == n:
                Popen(['sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)

        stream.stop_stream()
        buffer.seek(0)

    # undersample the signal to 16000 Hz
        audio = signal.resample_poly(np.frombuffer(buffer.getvalue(), dtype=np.int16), UP, DOWN) # the resampling is UP/DOWN of the original sampling rate
    audio = tf.cast(audio, dtype=tf.float32)

    # compute the STF
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # compute mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :output_fts]

    new_file_name = output_file + '/mfccs' + str(sample) + '.bin'
    mfcc_serialized = tf.io.serialize_tensor(mfccs)
    tf.io.write_file(new_file_name, mfcc_serialized)
    
    t_stop = time.time()
    print(f"Complete time: {t_stop - t_start}")
    

stream.close()    
p.terminate()
#Print the total time for the different VF levels
Popen(['cat /sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state'], shell=True)
Popen(['sudo sh -c "echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"'], shell=True)