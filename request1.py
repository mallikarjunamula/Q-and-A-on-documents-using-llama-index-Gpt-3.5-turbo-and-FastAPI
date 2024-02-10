import requests
import json

url = 'http://127.0.0.1:8000/qa_rag'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "paths": ['C:/Users/hp/Downloads/SuccessReceipt (1).pdf'],
    "query": "What is the transaction amount receipt?"
}
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json()['response'])