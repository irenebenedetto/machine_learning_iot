from DoSomething import DoSomething
import time
import argparse
import json
import tensorflow as tf
import datetime
import base64


ENCODING = 'utf-8'
class Inference(DoSomething):
	def __init__(self,clientID,publisher):
		super.__init__(self,clientID)
		self.interpreter = tf.lite.Interpreter(model_path=path)
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.publisher = publisher

	def notify(self, topic, msg):
		# message = {
        #     "bn": "pi@raspberrypi",
        #     "bt": int(datetime.datetime.now().timestamp()),
        #     "e": [
        #         {"n": "mfcc", "u": "/", "t": 0, "v": mfcc_string},
		# 		{"n": "it", "u": "/", "t": 0, "v": it}
        #     ]
        # }
		data = json.loads(msg.payload.decode()) 
		mfcc_string = data['e'][0]['v']
		it = data['e'][1]['v']
		#base64.b64encode(mfcc).decode(ENCODING)
		mfcc = base64.b64decode(mfcc_string.encode(ENCODING))
		self.interpreter.set_tensor(self.input_details[0]['index'], [mfcc])
		self.interpreter.invoke()
		logits = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

		response = {
            "bn": "modeln",
            "bt": int(datetime.datetime.now().timestamp()),
            "e": [
                {"n": "logits", "u": "/", "t": 0, "v": logits}
            ]
        }
		response = json.dumps(response)
		self.publisher.myMqttClient.myPublish (f"/277959/result/{it}", (response))




	


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', required=True, help='model version')
	args = parser.parse_args()
	path = args.version.lower()
    
	pub = DoSomething("publisher_inference")
	pub.run()
	sub = Inference("subscriber_inference",pub)
	sub.run()
	pub.myMqttClient.mySubscribe("/277959/mfcc")


	while(True):
		time.sleep(1)
		
	sub.end()
	pub.end()