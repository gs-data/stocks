import pymongo
import requests
import json

"""
Connect to Twitter API 1% sample stream
Open txt file to write all streamed data
Connect to local mongodb server to store data filtered by search_terms
"""

client = pymongo.MongoClient('localhost', 27017)
db = client.twitter
collection = db.aapl

search_terms = ['aapl', 'apple', 'stock']
path_to_raw_data = './data/raw/twitter_stream.txt'

with open('./secrets/twitter.txt', 'r') as secrets:
    for line in secrets:
        if 'Bearer token' in line:
            bearer_token = secrets.readline().strip()
headers = {"Authorization": f"Bearer {bearer_token}"}
            
url = "https://api.twitter.com/2/tweets/sample/stream"
query_params = {
    "tweet.fields": ",".join(["created_at", "lang"]),
    "expansions": "author_id",
    "user.fields": "created_at"
}

with open(path_to_raw_data, 'w') as data_file:
    with requests.get(url, params=query_params, headers=headers, stream=True) as response:
        print("Status code:", response.status_code)
        for response_line in response.iter_lines():
            if response_line:
                json_line = json.loads(response_line)
                if json_line['data']['lang'] == 'en':
                    data_file.write(response_line.decode('utf-8') + '\n')
                if any(s in json_line['data']['text'] for s in search_terms):
                    print(json_line['data']['text'])
                    collection.insert_one(json_line)
        if response.status_code != 200:
            raise Exception(
                "Request returned an error: {} {}".format(
                    response.status_code, response.text
                )
            )
