"""
Author : Rohan Das
Date : 09/09/21
"""

import os
import utils.config as config
import tensorflow as tf
from utils import model
import utils.data_management as dm
import time
import pandas
from utils import modelTuning


def get_unique_model_name(specific_name="Keras_Tuner_model"):
    model_fileName = time.strftime(f"{specific_name}_at_%Y%m%d_%H%M%S.h5")
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    model_file_path = os.path.join(config.TRAINED_MODEL_DIR, model_fileName)
    return model_file_path


def train():
    my_model = modelTuning.modelTuner()
    callbacks = model.callbacks()
    train_generator, valid_generator = dm.train_valid_data_gen()

    my_model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    model_file_path = get_unique_model_name()
    my_model.save(model_file_path)
    print(f"model saved at the following location\n ==>{model_file_path}")


if __name__ == "__main__":
    train()
