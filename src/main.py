import os
import time
from prediction import *
from http_requests import *
from dotenv import load_dotenv


def main(model_name, prediction_interval):
    load_dotenv()
    url = os.getenv("API_URL")

    test = True
    while test:
        get_response = get_data(url)
        if(get_response != None) :
            combined_data, window_id = combine_data_from_get_response(get_response)
            prediction = create_prediction(combined_data, model_name)
            post_prediction(url, prediction, window_id)
            time.sleep(prediction_interval)
        test = False


if __name__ == "__main__":
    main(model_name="model_v4_S2_60s_stress_mul8.keras", prediction_interval=30)
