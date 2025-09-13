import requests

url = "http://localhost:8000/predict"

with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
