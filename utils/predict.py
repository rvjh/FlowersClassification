

import os
import utils.config as config
import tensorflow as tf
import utils.data_management as dm
from PIL import Image
import numpy  as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array


class Predict:
    def __init__(self, latest=True, model_index=None):
        if latest:
            self.get_latest_model_path()
        elif (model_index is not None) and (not latest):
            self.model_index = model_index
            self.get_other_models()
        self.my_model = tf.keras.models.load_model(self.latest_model_path)

    def get_latest_model_path(self):
        available_models = os.listdir(config.TRAINED_MODEL_DIR)
        latest_model = sorted(available_models)[-1]
        self.latest_model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model)

    def get_other_models(self):
        available_models = os.listdir(config.TRAINED_MODEL_DIR)
        latest_model = sorted(available_models)[self.model_index]
        self.latest_model_path = os.path.join(config.TRAINED_MODEL_DIR, latest_model)

    def predict(self, input_img_path=None):
        image = load_img(input_img_path)
        #img = Image.open(input_img_path)
        image = dm.manage_input_data(image)
        #image = img_to_array(image)
        image = image / 255.0
        prediction_image = np.array(image)
        result = self.my_model.predict(prediction_image)
        value = np.argmax(result)
        #print("### RESULT: ", result)
        return value

    def mapping_folder(self,value):
        Name = []
        for file in os.listdir(r"D:\Practice\FlowersClassification\data\flowers"):
            Name += [file]
        print("Categories present: ", Name)

        N = []
        for i in range(len(Name)):
            N += [i]
        mapping = dict(zip(Name, N))
        reverse_mapping = dict(zip(N, Name))

        move_name=reverse_mapping[value]
        print("Prediction is {}.".format(move_name))




if __name__ == "__main__":
    obj = Predict()
    a = obj.predict(input_img_path='qwer.jpg')
    obj.mapping_folder(a)

