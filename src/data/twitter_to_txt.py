import requests
import json
from datetime import datetime, timezone

"""
Connect to Twitter API 1% sample stream
Write to txt file all streamed data filtered by lang == 'en'
  filenames: "YYYY-mm-dd-twitter-stream-XXX.txt"
"""

path_to_raw_data_pattern = './data/raw/{}-twitter-stream-{:03}.txt'

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

date_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

count = 0
batches = 0
with requests.get(url, params=query_params, headers=headers, stream=True) as response:
    print("Status code:", response.status_code)
    print("Headers: ", response.headers)
    print("x-rate-limit-remaining:", response.headers['x-rate-limit-remaining'])
    lines = response.iter_lines()
    while count < 15000:
        path_to_raw_data = path_to_raw_data_pattern.format(date_today, batches)
        with open(path_to_raw_data, 'w') as data_file:
            print(f"Writing to {path_to_raw_data}")
            for line in lines:
                if line:
                    json_line = json.loads(line)
                    if json_line['data']['lang'] == 'en':
                        data_file.write(line.decode('utf-8') + '\n')
                        count += 1
                        if (count % 5000 == 0):
                            break
            batches += 1
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
