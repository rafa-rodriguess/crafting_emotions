import requests
from pymongo import MongoClient

def send_discord_message(webhook_url: str, message: str, username: str = "Bot", avatar_url: str = None):
    data = {
        "content": message,
        "username": username
    }
    
    if avatar_url:
        data["avatar_url"] = avatar_url
    
    response = requests.post(webhook_url, json=data)
    
    if response.status_code == 204:
        print("Mensagem enviada com sucesso!")
    else:
        print(f"Erro ao enviar mensagem: {response.status_code} - {response.text}")