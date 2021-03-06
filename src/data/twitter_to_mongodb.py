import pymongo
import requests
import json

"""
Connect to Twitter API 1% sample stream
Connect to local mongodb server to store data
Filter stream by lang=='en' and write 5000 tweets to mongo
"""

client = pymongo.MongoClient('localhost', 27017)
db = client.twitter
collection = db.raw

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

count = 0
with requests.get(url, params=query_params, headers=headers, stream=True) as response:
    print("Status code:", response.status_code)
    print("Headers: ", response.headers)
    print("x-rate-limit-remaining:", response.headers['x-rate-limit-remaining'])
    for line in response.iter_lines():
        if line:
            json_line = json.loads(line)
            if json_line['data']['lang'] == 'en':
                collection.insert_one(json_line)
                count += 1
                if count == 5000:
                    break     
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
