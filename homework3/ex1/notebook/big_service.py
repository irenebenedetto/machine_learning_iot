import base64
import io
import json
import cherrypy
import datetime
import tensorflow as tf
import numpy as np
from scipy import signal
import wave
import os

class BigModelGenerator(object):
    exposed = True


    def POST(self):

        body = cherrypy.request.body.read()
        body = json.loads(body)

        audio = body["e"][0]["vd"]
        audio = base64.b64decode(audio.encode())

        audio = np.frombuffer(audio, dtype=np.float32)

        sampling_rate = 16000
        frame_length, frame_step = 640, 320
        lower_frequency, upper_frequency = 20, 4000

        num_coefficients = 30
        num_mel_bins = 40

        num_spectrogram_bins = (frame_length) // 2 + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency, upper_frequency)

        # creating mfccs

        zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([sampling_rate])

        stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
        spectrogram = tf.abs(stft)

        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :num_coefficients]
        mfccs = tf.expand_dims(mfccs, -1)

        interpreter = tf.lite.Interpreter(model_path=f'./big.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], [mfccs])
        interpreter.invoke()
        y_pred = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0]).tolist()

        response = {
            "bn": "big_model@127.0.0.1",
            "bt": int(datetime.datetime.now().timestamp()),
            "e": [
                {"n": "label", "u": "/", "t": 0, "v": y_pred}
            ]
        }
        return json.dumps(response)



if __name__ == '__main__':
    
    if 'data' not in os.listdir():
        zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')
        
    conf = {
            '/': {
                'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                'tools.sessions.on': True,
            }
    }
    cherrypy.tree.mount(BigModelGenerator(), '/big_model', conf)
    cherrypy.config.update({'server.socket_host': '127.0.0.1'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
