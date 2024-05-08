import time
from inference import *
from stress_predictor.src.influx import *
from dotenv import load_dotenv


def main(model_name):
    load_dotenv()
    url = os.getenv("INFLUX_URL")

    while True:
        get_response = get_data(url)
        combined_data = combine_data_from_get_response(get_response)
        prediction = create_predictions(combined_data, model_name)
        post_prediction(url, prediction)
        time.sleep(5)


if __name__ == "__main__":
    main(model_name="model_v3_S2_120s.keras")
