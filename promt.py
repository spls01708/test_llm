import requests

url = "http://0.0.0.0:8080/query"

data = {
            "question" : "การปลดข้าราชการพลเรือน"
    }

response = requests.post(url, json=data)
print("Response:", response.json())


