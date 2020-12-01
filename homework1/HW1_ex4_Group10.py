import datetime
import time
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from os import path
import argparse 

parser=argparse.ArgumentParser()
parser.add_argument('--input', help='input file')
parser.add_argument('--output', help='output file')

args = parser.parse_args()


input_path = args.input
output_file = args.output

tot_size = path.getsize(input_path)

# open the TFRecordWriter class
with tf.io.TFRecordWriter(output_file) as writer:
  with open(input_path + "/samples.csv") as input_file:
    lines = csv.reader(input_file)
    for fields in lines:
      
      date = fields[0]
      hour = fields[1]
      temp = int(fields[2])
      humidity = int(fields[3])

      path_audio = input_path + "/" + fields[4]
      tot_size += path.getsize(path_audio)

      # convert date into posix int 
      dt = datetime.datetime.strptime(date + " " + hour, "%d/%m/%Y %H:%M:%S")
      posix = int(time.mktime(dt.timetuple()))

      # create the features 
      posix_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[posix])) 
      temp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[temp])) 
      humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[humidity])) 

      # read audio file
      raw_audio = tf.io.read_file(path_audio)
      # audio, _ = tf.audio.decode_wav(raw_audio)

      audio_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_audio.numpy()]))
      #audio_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[audio]))

      # crate the dictionary for the structure of each record
      mapping = {
          
          'datetime': posix_feature, 
          'temperature': temp_feature,
          'humidity': humidity_feature,
          'audio': audio_feature,

      }
      example = tf.train.Example(features=tf.train.Features(feature=mapping)) 
      writer.write(example.SerializeToString())

red_size = path.getsize(output_file)

print(f"From {str(tot_size/(1024.*1024.))} to {str(red_size/(1024.*1024.))} Mb")




