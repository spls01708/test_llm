import requests
from rich import print
from rich.console import Console

console = Console()

url = "http://0.0.0.0:8080/query"

data = {
            "question" : "การปลดข้าราชการพลเรือน"
    }

r = requests.post(url, json=data)
response = r.json()
# print("Response:", response.json())

console.print("[bold green]Answer:[/bold green]")
console.print(response["answer"], width=80)

console.print("\n[bold cyan]Sources:[/bold cyan]")
for source in response["sources"]:
    console.print(f"- {source}")


