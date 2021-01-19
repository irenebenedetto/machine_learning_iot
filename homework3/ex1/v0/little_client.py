import argparse
import os
import numpy as np
import os
import tensorflow as tf
import zlib
import tensorflow_model_optimization as tfmot
from scipy.signal import resample_poly
import base64
import json
import tqdm
import datetime
import requests

frame_length, frame_step = 256, 128
sampling_rate = 16000




def read(file_path, labels):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label_id = tf.argmax(label == labels)
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)

    audio = tf.squeeze(audio, axis=1)

    return audio, label_id

def pad(audio):
    zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    audio = tf.concat([audio, zero_padding], 0)
    audio.set_shape([sampling_rate])

    return audio

def get_spectrogram(audio):
    stft = tf.signal.stft(audio, frame_length=frame_length,
            frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)

    return spectrogram




if __name__ == '__main__':
    ENCODING = 'utf-8'
    communication_cost = 0.0
    LABELS = np.loadtxt('./labels.txt', dtype=str)
    sampling_rate = 16000

    with open('./small.tflite.zlib', 'rb') as fp:
        model_zip = zlib.decompress(fp.read())
        interpreter = tf.lite.Interpreter(model_content=model_zip)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = interpreter.get_tensor(output_details[0]['index'])

    test_files = open('./kws_test_split.txt', 'r').read().splitlines()

    theta = 0.05
    accuracy = 0

    for it, file_path in enumerate(tqdm.tqdm(test_files)):
        audio, y_true = read(file_path, LABELS)
        audio_padded = pad(audio)

        spectrogram = get_spectrogram(audio_padded)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [56, 56])
        
        interpreter.set_tensor(input_details[0]['index'], [spectrogram])

        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])[0]
        y_pred = y_pred / np.linalg.norm(y_pred)
        list_sm = tf.sort(y_pred, direction='DESCENDING')[:2]
        sm = list_sm[0] - list_sm[1]

        if sm < theta:
            audio_bytes = base64.b64encode(audio)
            audio_string = audio_bytes.decode(ENCODING)

            request = {
                "bn": "127.0.0.1",
                "bt": int(datetime.datetime.now().timestamp()),
                "e": [
                    {"n": "a", "u": "/", "t": 0, "vd": audio_string}
                ]
            }
            request = json.dumps(request)

            communication_cost += len(request)
            print(f'Total communication cost: {communication_cost/(1024*1024)}')
            r = requests.post('http://127.0.0.1:8080/big_model', request)
            if r.status_code == 200:
                y_pred = r.json()['e'][0]['v']
            else:
                print('Error with the big model prediction')
        else:
            y_pred = tf.argmax(y_pred)

        if y_pred == y_true:
            accuracy +=1
            print(f'Accuracy: {accuracy/(it+1)}')

    accuracy = round(accuracy/len(test_files)*100, 3)
    print(f'Accuracy: {accuracy}%')
    print(f'Communication Cost: {communication_cost/(1024*1024)} MB')




