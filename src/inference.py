import os
import keras
import numpy as np


def combine_data_from_get_response(get_response):
    bvp_list = []
    eda_list = []
    temp_list = []

    # for table in get_response:
    #     for record in table.records:
    #         if record.get_field() == "bvp":
    #             bvp_list.append(record.get_value())
    #         elif record.get_field() == "eda":
    #             eda_list.append(record.get_value())
    #         elif record.get_field() == "temp":
    #             temp_list.append(record.get_value())

    data_list = bvp_list + eda_list + temp_list
    data_array = np.array(data_list)
    return data_array


class SliceLayer(keras.layers.Layer):
    def __init__(self, start_index, end_index, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index

    def call(self, inputs):
        return inputs[:, self.start_index : self.end_index]

    def get_config(self):
        config = super(SliceLayer, self).get_config()
        config.update({"start_index": self.start_index, "end_index": self.end_index})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_predictions(data, model_name):
    model = keras.models.load_model(filepath=os.path.join("models", model_name), custom_objects={"SliceLayer": SliceLayer})
    prediction = model.predict(x=data, verbose=1)
    return prediction
