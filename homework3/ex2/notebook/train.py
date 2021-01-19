import argparse
import os
import numpy as np
import os
import tensorflow as tf
import zlib
import tensorflow_model_optimization as tfmot
from scipy.signal import resample_poly


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False, resampling=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.resampling = resampling
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)
        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)
        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [56, 56])
        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        if self.resampling:
            audio = resample_poly(audio, self.sampling_rate, 16000)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds


class MyModel:
    def __init__(self, model_name, alpha, input_shape, output_shape, version, final_sparsity=None):

        if use_mfccs:
            strides = [2, 1]
        else:
            strides = [2, 2]
        if model_name == '1':
           model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(128 * alpha),
                                                                kernel_size=[3, 3], strides=strides, use_bias=False),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(alpha * 512),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(alpha * 512),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),

                                         tf.keras.layers.GlobalAvgPool2D(),
                                         tf.keras.layers.Dense(512),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(256),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(128),
                                         tf.keras.layers.Dense(output_shape)])
           
        elif model_name =='2':
             model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(input_shape = input_shape, filters=128, kernel_size=[3,3], strides=strides,use_bias=False,activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=[1, 1],use_bias=False,),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3],strides=[1, 1],use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(output_shape),
            ])
          
        elif model_name == '3':
            model = tf.keras.applications.MobileNet(input_shape=input_shape,classes = output_shape, classifier_activation=None,weights = None)
            
          
        elif model_name == '4':
            model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                                kernel_size=[3, 3], strides=strides, use_bias=False,
                                                                name='first_conv1d'),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                                name='second_conv1d'),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(alpha * 256),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                                name='third_conv1d'),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.GlobalAvgPool2D(),
                                         tf.keras.layers.Dense(32, name='fc1'),
                                         tf.keras.layers.Dense(output_shape, name='fc2')])
            
            
        elif model_name == '5':
            model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                                kernel_size=[3, 3], strides=strides, use_bias=False,
                                                                name='first_conv1d'),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                                name='second_conv1d'),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(alpha * 256),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                                name='third_conv1d'),
                                         tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                         use_bias=False, ),
                                         tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(alpha * 128),
                                                                kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                                name='fourth_conv1d'),
                                         tf.keras.layers.BatchNormalization(momentum=0.1),
                                         tf.keras.layers.ReLU(),
                                         tf.keras.layers.GlobalAvgPool2D(),
                                         tf.keras.layers.Dense(18, name='fc1'),
                                         tf.keras.layers.Dense(output_shape, name='fc2')])
        model.summary()
        self.model = model
        self.alpha = alpha
        self.final_sparsity = final_sparsity
        self.model_name = model_name.lower()
        self.version = version.lower()
        if alpha != 1:
            self.model_name += '_ws' + str(alpha).split('.')[1]
        if final_sparsity is not None and 'lstm' not in self.model_name:
            self.model_name += '_mb' + str(final_sparsity).split('.')[1]
            self.magnitude_pruning = True
        else:
            self.magnitude_pruning = False

        self.final_sparsity = final_sparsity
        self.input_shape = input_shape
        

    def compile_model(self, optimizer, loss_function, eval_metric):
        if self.magnitude_pruning:
            # sparsity scheduler
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.1,
                    final_sparsity=self.final_sparsity,
                    begin_step=len(train_ds) * 5,
                    end_step=len(train_ds) * 25)
            }

            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            self.model = prune_low_magnitude(self.model, **pruning_params)

            input_shape = [32] + self.input_shape
            self.model.build(input_shape)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=eval_metric
        )

    def train_model(self, X_train, X_val, N_EPOCH, callbacks=[]):
        if self.magnitude_pruning:
            callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
            
        print('\tTraining... ')
        print('\t', end='')
        history = self.model.fit(
            X_train,
            epochs=N_EPOCH,
            validation_data=X_val,
            verbose=1,
            callbacks=callbacks,
        )

        return history

    def convert_to(self):
        # --------- with tflite quantization weight only
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # convert the model into a tflite version
        tflite_model = converter.convert()

        with open(f'./{self.version}.tflite', 'wb') as fp:
            fp.write(tflite_model)


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True, help='model version')
    args = parser.parse_args()

    version = args.version.lower()

    def scheduler(epoch, lr):
        if epoch == 20 or epoch == 25:
            return lr * 0.1
        else:
            return lr

    model_name = version
    sampling_rate = 16000
    use_mfccs = True
    OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
              'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
              'num_coefficients': 32}
    final_sparsity = None
    N_EPOCH = 32
    LR = 0.01
    alpha = 1.3

    callbacks = [
      tf.keras.callbacks.LearningRateScheduler(schedule=scheduler),
      tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=30,
                                        restore_best_weights=True)
    ]

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    train_files = open('./kws_train_split.txt', 'r').read().splitlines()
    val_files = open('./kws_val_split.txt', 'r').read().splitlines()
    test_files = open('./kws_test_split.txt', 'r').read().splitlines()
    train_files = tf.convert_to_tensor(train_files)
    val_files = tf.convert_to_tensor(val_files)
    test_files = tf.convert_to_tensor(test_files)

    LABELS = np.loadtxt('./labels.txt', dtype=str)

    generator = SignalGenerator(LABELS, sampling_rate=sampling_rate, **OPTIONS)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    for x, y in train_ds:
        input_shape = x.shape.as_list()[1:]
        output_shape = y.shape.as_list()[1:]
        break

    print(f'Input shape: {input_shape}')

    output_shape = 8

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    eval_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model = MyModel(model_name, alpha, input_shape, output_shape,version, final_sparsity)
    model.compile_model(optimizer, loss_function, eval_metric)
    history = model.train_model(train_ds, val_ds, N_EPOCH,
                                callbacks=callbacks)

    model.convert_to()