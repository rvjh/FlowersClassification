import os
import utils.config as config
from utils import model
import keras_tuner as kt
import utils.data_management as dm



def modelTuner():
    #model_tu = model.build_model()
    tuner = kt.RandomSearch(model.build_model, objective='val_loss', max_trials=3)
    print("Summary of tuning: ", tuner.search_space_summary())
    training_set, validation_set = dm.train_valid_data_gen()
    report1 = tuner.search(training_set, epochs=3, validation_data=validation_set)
    print(f"Report is\n: {report1}")

    full_model = tuner.get_best_models(num_models=2)[0]
    print("custom model summary")
    full_model.summary()

    return full_model

