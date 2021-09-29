import os
import tensorflow as tf
import utils.config as config
import time
import keras_tuner as kt
from tensorflow import keras


def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=config.IMAGE_SIZE
    ))
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("base_model is loaded")
    model.summary()
    return model

'''
def tune_model():
    model = build_model()
    tuner = kt.RandomSearch(model, objective='val_loss', max_trials=2)
    tuner.search(training_set, epochs=2, validation_data=validation_set)
    full_model = tuner.get_best_models(num_models=1)[0]
    print("custom model summary")
    full_model.summary()

    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return full_model
'''

def callbacks(base_dir="."):
    # tensorboard callbacks
    base_log_dir = config.TENSORBOARD_ROOT_LOG_DIR
    unique_log = time.strftime("log_at_%Y%m%d_%H%M%S")
    tensorboard_log_dir = os.path.join(base_log_dir, unique_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

    # checkpoint callbacks

    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, "Keras_Tuner_model_checkpoint.h5")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_best_only=True
    )

    callback_list = [tensorboard_cb, checkpoint_cb]

    return callback_list


if __name__ == "__main__":
    #build_model()
    tune_model()
