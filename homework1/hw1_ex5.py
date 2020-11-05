"""
parser=argparse.ArgumentParser()
parser.add_argument('-num_sample',type=float, help='input file folder')
parser.add_argument('-output',type=float, help='period in seconds')


"""


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

num_sample = int(sys.argv[1])
output_file = sys.argv[2]

output_fts, num_mel_bins, lower_frequency, upper_frequency = 10, 40, 20, 4000
UP, DOWN = 1, 3


RATE, CHANNELS, CHUNK_SIZE, L, FORMAT = 48000, 1, 1024, 1, pyaudio.paInt16

for sample in range(num_sample):

    p = pyaudio.PyAudio()
    stream = p.open(format = FORMAT, channels = CHANNELS, rate = RATE, input=True)

    frames = []
    for i in range(int(RATE/CHUNK_SIZE)*L):
            # NUMBER OF SAMPLES IN A BLOCK OF DATA
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
        
    stream.stop_stream()
    stream.close()
    p.terminate()

    """
    buffer = io.BytesIO()
    wf = wave.open(buffer, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    """
    
    #_, audio = wavfile.read(io.BytesIO(b''.join(frames)))
    t1 = time.time()
    # for sample rate conversion to reduce the memory requirements and accelerate the subsequent processing steps
    #rate, audio = wavfile.read(buffer)
    audio = signal.resample_poly(audio, UP, DOWN) # the resampling is UP/DOWN of the original sampling rate
    #audio = audio.astype(np.int16)
    audio = tf.convert_to_tensor(audio)
    
    print(type(audio))
    print(audio.shape)
    #spectrogram = tf.io.parse_tensor(audio, out_type=tf.float32)
    #tf_audio = tf.squeeze(audio, 1)
    
    frame_length = int(16000 * 0.04) #???
    frame_step = int(16000 * 0.02)
    
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    
    spectrogram = tf.abs(stft) 
    
    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, RATE, lower_frequency, upper_frequency)

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :output_fts]

    image = tf.transpose(mfccs)
    # added a dimension for the channel rgb
    image = tf.expand_dims(image, -1)
    
    # convert to log 
    image = tf.math.log(image + 1e-6)
    # normalize image
    min_ = tf.reduce_min(image)
    max_ = tf.reduce_max(image)
    image = (image - min_) / (max_ - min_) 
    image = image * 255.
    # to int --> ?
    image = tf.cast(image, tf.uint8)

    # to png 
    image_png = tf.io.encode_png(image)

    new_file_name = output_file + str(sample) + '.png'
    tf.io.write_file(new_file_name, image_png)
    
    preproc_time = time.time() - t1
    
    print(f"Preprocessing time: {preproc_time}")
    
    print(f'Image "{new_file_name}" saved!')

    size = path.getsize(new_file_name)
    print(f'File "{new_file_name}" of size {str(round(size/(8*1024), 3))} Kbyte.')
 
