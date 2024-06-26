import os
import keras
import numpy as np


def combine_data_from_get_response(get_response):
    """
    Combines data from the GET response into a single array and extracts the window ID.

    Args:
        get_response: The response object containing the data.

    Returns:
        numpy.ndarray: Combined data array.
        int: The window ID.
    """
    bvp_list = []
    eda_list = []
    temp_list = []

    for row in get_response.json():
        if row["data_type"] == "bvp":
            bvp_list.append(row["_value"])
        elif row["data_type"] == "eda":
            eda_list.append(row["_value"])
        else:
            temp_list.append(row["_value"])

    data_array = np.concatenate([np.array(bvp_list), np.array(eda_list), np.array(temp_list)])
    return data_array, int(get_response.json()[0]["window_id"])


class SliceLayer(keras.layers.Layer):
    """
    Custom Keras layer for slicing input tensors.

    Args:
        start_index (int): Index to start slicing from.
        end_index (int): Index to end slicing.

    Returns:
        Tensor: Sliced tensor.
    """

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


def create_prediction(data, model_name):
    """
    Loads a trained Keras model and generates predictions for the given data.

    Args:
        data (numpy.ndarray): Input data for prediction.
        model_name (str): Name of the model file.

    Returns:
        numpy.ndarray: Predictions generated by the model.
    """
    model = keras.models.load_model(filepath=os.path.join("models", model_name), custom_objects={"SliceLayer": SliceLayer})
    data = data.reshape(-1, 1)[np.newaxis, :]
    prediction = int(np.round(model.predict(x=data, verbose=1)[0][0]))
    return prediction
