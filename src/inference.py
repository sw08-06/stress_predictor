import os
import keras
import numpy as np
import requests
import time
#from influxdb_client import InfluxDBClient

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

# Set up global variables
url = os.getenv('URL')
token = os.getenv('TOKEN')
org = os.getenv('ORG')
bucket = os.getenv('BUCKET')#måske ik nødvendigt
window_id = 0

#predict on data
def model_inference(data, model):
    prediction = model.predict(x=data, verbose=1)
    return prediction

#make GET request to API
def request_data(bucket, window_id):
    url = 'http://localhost:3000/api/stress_predict'#fix så det er rigtigt endpoint
    params = {
    'string_param': bucket,#måske ik nødvendigt
    'string_param': "data",
    'number_param': window_id
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response#fluxobject eller json
    else:
        print('Error: ', response.status_code)
        return 0

#make POST request to API
def post_preds(data, model, window_id):
    url = 'http://localhost:3000/api/stress_predict'#fix så det er rigtigt endpoint
    prediction=np.round(model_inference(data, model))
    send_data = {
        'string_param': "prediction",
        'number_param': prediction,
        'number_param': window_id
    }
    response = requests.post(url, data = send_data)
    if response.status_code == 200:
        print(f"Successfully posted prediction for window_id: {window_id}")
    else:
        print('Error: ', response.status_code)
    
# def query_data():
#     url = os.getenv('URL')
#     token = os.getenv('TOKEN')
#     org = os.getenv('ORG')
#     bucket = "bucket_name"
#     measurement = "data"
#     window_id = 2#set up window_id so it increases by one each time
#     client = InfluxDBClient(url=url, token=token, org=org)

#     # Create the query API
#     query_api = client.query_api()

#     # Retrieve all data from measurement in bucket
#     flux_query = f'from(bucket: "{bucket}") 
#                 |> range(start: -inf) 
#                 |> filter(fn: (r) => r._measurement == "{measurement}")
#                 |> filter(fn: (r) => r.window_id == "{window_id}")'

#     result = query_api.query(flux_query)
#     return result

#concatenate data from fluxobject
def build_array(data):#sat op til fluxobject
    bvp_list = []
    eda_list = []
    temp_list = []

    # Iterate and extract values for each tag
    for table in data:
        for record in table.records:
            if record.get_field() == 'bvp':
                bvp_list.append(record.get_value())
            elif record.get_field() == 'eda':
                eda_list.append(record.get_value())
            elif record.get_field() == 'temp':
                temp_list.append(record.get_value())

    data_list = bvp_list + eda_list + temp_list
    data_array = np.array(data_list)
    return data_array

#continously run the inference with a model, making GET and POST requests to the API
if __name__ == "__main__":
    window_id = 0
    model = keras.models.load_model(filepath="src/model_v3_S2_120s.keras", custom_objects={"SliceLayer": SliceLayer})
    while True:
        data = request_data(bucket, window_id)
        if data == 0:
            print("No new data")
        else:
            data_array = build_array(data)
            post_preds(data_array, model, window_id)
            window_id += 1
        time.sleep(30)
