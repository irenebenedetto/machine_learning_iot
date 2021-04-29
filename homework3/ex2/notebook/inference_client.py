from DoSomething import DoSomething
import time
import argparse
import json
import tensorflow as tf
import datetime
import base64
import numpy as np
import zlib

ENCODING = 'utf-8'
class Inference(DoSomething):
	def __init__(self,clientID,publisher):
		super(Inference,self).__init__(clientID)
		if 'zlib' in path:
			with open(path, 'rb') as fp:
				m = zlib.decompress(fp.read())
				self.interpreter = tf.lite.Interpreter(model_content=m)
		else:
			self.interpreter = tf.lite.Interpreter(model_path=path)

		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.publisher = publisher

	def notify(self, topic, msg):
		print ("Received '%s' with topic '%s'" % ('mfcc', topic))
		data = json.loads(msg) 
		mfcc_string = data['e'][0]['v']
		it = data['e'][1]['v']
		mfcc = base64.b64decode(mfcc_string.encode())
		mfcc = np.frombuffer(mfcc, dtype=np.float32)
		shape = data['e'][2]['v']
		mfcc = tf.reshape(mfcc, shape)
		self.interpreter.set_tensor(self.input_details[0]['index'], [mfcc])
		self.interpreter.invoke()
		logits = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

		response = {
                    "bn": "modeln",
                    "bt": int(datetime.datetime.now().timestamp()),
                    "e": [{"n": "logits", "u": "/", "t": 0, "v": logits.tolist()}]
                }
		response = json.dumps(response)
		print ("publishing '%s' with topic '%s'" % ('logits', f"/Team10/277959/result"))
		self.publisher.myMqttClient.myPublish (f"/Team10/277959/result", (response))
		

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', required=True, help='model version')
	args = parser.parse_args()
	path = args.model.lower()
    
	pub = DoSomething(f"publisher_inference_{path}")
	pub.run()
	sub = Inference(f"subscriber_inference_{path}",pub)
	sub.run()
	sub.myMqttClient.mySubscribe("/Team10/277959/mfcc")


	while(True):
		time.sleep(1)
		
	sub.end()
	pub.end()
