import requests
from rich import print
from rich.console import Console

console = Console()

url = "http://0.0.0.0:8080/query"

data = {
    "question": "คนพิการเป็นข้าราชการพลเรือนได้ไหม"
}

r = requests.post(url, json=data)
response = r.json()

# แสดงคำถามด้วยสีแดง
console.print("[bold red]Question:[/bold red]")
console.print(data["question"], width=80)

# แสดงคำตอบด้วยสีเขียว
console.print("\n[bold green]Answer:[/bold green]")
console.print(response["answer"], width=80)

# แสดงแหล่งที่มาด้วยสีฟ้า
console.print("\n[bold cyan]Sources:[/bold cyan]")
for source in response["sources"]:
    console.print(f"- {source}")


