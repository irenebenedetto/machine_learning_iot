import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import zlib


class WindowGenerator:
    def __init__(self, input_width, mean, std):
        self.input_width = input_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :6, :]
        labels = features[:, -6:, :]
        inputs.set_shape([None, 6, 2])
        labels.set_shape([None, 6, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train, batch_size=32):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=12,
            sequence_stride=1,
            batch_size=batch_size)

        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds.prefetch(tf.data.experimental.AUTOTUNE)


class MyModel:
    def __init__(self, model_name, alpha, input_shape, output_shape,version, batch_size=32, final_sparsity=None):

        if model_name.lower() == 'mlp_b':
            # create the mlp model
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=input_shape, name='flatten'),
                tf.keras.layers.Dense(int(alpha * 128), activation='relu', name='first_dense'),
                tf.keras.layers.Dense(int(alpha * 128), activation='relu', name='second_dense'),
                tf.keras.layers.Dense(12, name='third_dense'),
                tf.keras.layers.Reshape(output_shape)

            ])

        elif model_name.lower() == 'mlp_a':
            # create the mlp model
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=input_shape, name='flatten'),
                tf.keras.layers.Dense(int(alpha * 128), activation='relu', name='first_dense'),
                tf.keras.layers.Dense(12, name='third_dense'),
                tf.keras.layers.Reshape(output_shape)

            ])


        model.summary()
        self.model = model
        self.alpha = alpha
        self.batch_size = batch_size
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

    def compile_model(self, optimizer, loss_function, eval_metric):

        if self.magnitude_pruning:
            # sparsity scheduler
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=self.final_sparsity,
                    begin_step=len(train_ds) * 5,
                    end_step=len(train_ds) * 15)
            }

            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            self.model = prune_low_magnitude(self.model, **pruning_params)

            input_shape = [self.batch_size, 6, 2]
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

    def evaluate_model(self, X_test):
        return self.model.evaluate(X_test)

    def prune_model(self, weights_only=True):

        self.model = tfmot.sparsity.keras.strip_pruning(self.model)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if weights_only:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(f'./Group10_th_{self.version}.tflite.zlib', 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)


class CustomMAE(tf.keras.metrics.Metric):
    def __init__(self, name='CustomMAE', **kwargs):
        super(CustomMAE, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(shape=[2], name='sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # accumulate at each batch
        values = tf.cast(tf.abs(y_true - y_pred), dtype=tf.float32)

        self.sum.assign_add(tf.reduce_mean(values, axis=[0, 1]))
        self.count.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.sum, self.count)

    def reset_states(self):
        self.sum.assign(tf.zeros_like(self.sum))
        self.count.assign(tf.zeros_like(self.count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, help='version')
    args = parser.parse_args()

    version = args.version.lower()
    if version == 'a':
        model_name = 'mlp_a'
        alpha = 0.3
        final_sparsity = 0.9
        N_EPOCH = 70
        LR = 0.1
        BATCH_SIZE = 512

        MILESTONE = [10, 50, 60]
        def scheduler(epoch, lr):
            if epoch in MILESTONE:
                return lr*0.1
            else:
                return lr


    else:
        alpha = 0.1
        final_sparsity = 0.85
        model_name = 'mlp_b'
        N_EPOCH = 20
        LR = 0.01
        BATCH_SIZE = 32

        def scheduler(epoch, lr):
            if epoch % 10 == 0:
                return lr
            else:
                return lr

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)

    column_indices = [2, 5]
    columns = df.columns[column_indices]
    data = df[columns].values.astype(np.float32)

    n = len(data)
    train_data = data[0:int(n * 0.7)]
    val_data = data[int(n * 0.7):int(n * 0.9)]
    test_data = data[int(n * 0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    input_width = 12
    generator = WindowGenerator(input_width, mean, std)
    train_ds = generator.make_dataset(train_data, True, BATCH_SIZE)
    val_ds = generator.make_dataset(val_data, False, BATCH_SIZE)
    test_ds = generator.make_dataset(test_data, False, BATCH_SIZE)

    # extracting paramenters for model creation
    for x, y in train_ds:
        input_shape = x.shape.as_list()[1:]
        output_shape = y.shape.as_list()[1:]
        break




    eval_metric = [CustomMAE()]
    loss_function = [tf.keras.losses.MeanSquaredError()]
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    
    # model_name, alpha, input_shape, output_shape, final_sparsity
    model = MyModel(model_name, alpha, input_shape, output_shape, version,BATCH_SIZE, final_sparsity)
    model.compile_model(optimizer, loss_function, eval_metric)
    history = model.train_model(train_ds, val_ds, N_EPOCH, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])

    # magnitude based pruning
    model.prune_model()
