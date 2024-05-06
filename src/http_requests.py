import requests


def get_data(url):
    try:
        response = requests.get(f"{url}/api/stress_predict")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print("GET request failed:", e)
        return None


def post_prediction(url, prediction):
    try:
        response = requests.post(f"{url}/api/stress_predict", data={"prediction": prediction})
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print("POST request failed:", e)
        return None
