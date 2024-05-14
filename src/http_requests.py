import time
import requests


def get_data(url):
    """
    Fetches data from the specified URL.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        requests.Response: The response object containing the fetched data, or None if the request fails.
    """
    try:
        response = requests.get(f"{url}/api/stress-predict")
        if response.status_code == 204:
            print("No new data available")
            return None
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        print("GET request failed:", e)
        return None


def post_prediction(url, prediction, window_id):
    """
    Sends a POST request with prediction data to the specified URL.

    Args:
        url (str): The URL to send the POST request to.
        prediction: The prediction data to be sent.

    Returns:
        requests.Response: The response object from the server, or None if the request fails.
    """
    try:
        response = requests.post(f"{url}/api/stress-predict", json={"time": time.time_ns(), "window_id": window_id, "prediction": prediction})
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print("POST request failed:", e)
        return None
