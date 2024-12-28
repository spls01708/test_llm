import requests

url = "http://0.0.0.0:8080/query"

data = {
            "question" : "ข้าราชการพลเรือนกับการรับของขวัญ"
    }

response = requests.post(url, json=data)
print("Response:", response.json())


