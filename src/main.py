import time
from dotenv import load_dotenv
from inference import *
from http_requests import *


def main(stress_prediction_interval, model_name):
    load_dotenv()
    url = os.getenv("INFLUX_URL")
    while True:
        get_response = get_data(url)
        combined_data = combine_data_from_get_response(get_response)
        prediction = create_predictions(combined_data, model_name)
        post_prediction(url, prediction)
        time.sleep(stress_prediction_interval)


if __name__ == "__main__":
    main(5, "model_v3_S2_120s.keras")
