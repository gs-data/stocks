import json

path_to_raw_data = './data/raw/twitter_stream.txt'
path_to_processed_data = './data/processed/twitter_texts.txt'

with open(path_to_raw_data, 'r') as raw:
    with open(path_to_processed_data, 'w') as processed:
        for line in raw:
            processed.write(json.loads(line)['data']['text'].replace('\n', '\\n') + '\n')
