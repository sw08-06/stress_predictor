import requests


def get_data(url):
    response = requests.get(f"{url}/api/stress_predict")
    if response.status_code == 200:
        return response
    else:
        print("GET request failed:", response.status_code)
        return None


def post_prediction(url, prediction):
    response = requests.post(f"{url}/api/stress_predict", data=prediction)
    if response.status_code == 200:
        return response
    else:
        print("POST request failed:", response.status_code)
        return None
