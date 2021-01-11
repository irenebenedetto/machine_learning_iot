from DoSomething import DoSomething
import time
import os 
import tensorflow as tf
import json
import base64
import numpy as np
import datetime

N = 2
ENCODING = 'utf-8'
sampling_rate = 16000
frame_length, frame_step = 640, 320
lower_frequency, upper_frequency = 20, 4000
num_coefficients = 30
num_mel_bins = 40

num_spectrogram_bins = (frame_length) // 2 + 1
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency, upper_frequency)

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

def read(file_path, labels):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label_id = tf.argmax(label == labels)
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)

    audio = tf.squeeze(audio, axis=1)

    return audio, label_id

def get_mfcc(spectrogram):
	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
	mfccs = mfccs[..., :num_coefficients]
	mfccs = tf.expand_dims(mfccs, -1)
	return mfccs

class PredictionReceiver(DoSomething):
	def __init__(self,clientID):
		super.__init__(self,clientID)
		self.predictions = [] #list of logits

	def notify(self, topic, msg):
		# manage here your received message. You can perform some error-check here 
		data = json.loads(msg.payload.decode()) 
		print ("received '%s' under topic '%s'" % (msg, topic))
		# response = {
        #     "bn": "modeln",
        #     "bt": int(datetime.datetime.now().timestamp()),
        #     "e": [
        #         {"n": "logits", "u": "/", "t": 0, "v": logits}
        #     ]
        # }
		self.predictions.append(data['e'][0]['v']) 

	def get_prediction(self):
		while(True):
			if len(self.predictions) == N:
				break
		y_pred = tf.zeros(self.predictions[0].shape)
		for logits in self.predictions:
			y_pred += tf.nn.softmax(logits)
		y_pred = y_pred/N
		prediction = tf.argmax(y_pred)
		self.reset()
		return prediction

	def reset(self):
		self.predictions = []
	

if __name__ == "__main__":

	if 'data' not in os.listdir():
		zip_path = tf.keras.utils.get_file(
			origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
			fname='mini_speech_commands.zip',
			extract=True,
			cache_dir='.', cache_subdir='data')

	test_files = open('./kws_test_split.txt', 'r').read().splitlines()
	LABELS = np.loadtxt('./labels.txt', dtype=str)

	pub = DoSomething("publisher")
	pub.run()
	sub = PredictionReceiver("subscriber")
	sub.run()
	

	for it, file_path in enumerate(test_files):
		sub.myMqttClient.mySubscribe(f"277959/result/{it}")
		audio, y_true = read(file_path, LABELS)
		audio_padded = pad(audio)
		spectrogram = get_spectrogram(audio_padded)
		mfcc = get_mfcc(spectrogram)
		mfcc_string = base64.b64encode(mfcc).decode(ENCODING)
		response = {
            "bn": "pi@raspberrypi",
            "bt": int(datetime.datetime.now().timestamp()),
            "e": [
                {"n": "mfcc", "u": "/", "t": 0, "v": mfcc_string},
				{"n": "it", "u": "/", "t": 0, "v": it}
            ]
        }
		message = json.dumps(response)
		pub.myMqttClient.myPublish ("/277959/mfcc", (message))
		y_pred = sub.get_prediction()
		
		if y_pred == y_true:
			accuracy +=1
		
	accuracy = round(accuracy/len(test_files)*100, 3)
	print(f"Accuracy: {accuracy}")
	pub.end()
	sub.end()