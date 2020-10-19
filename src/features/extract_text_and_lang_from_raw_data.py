import json
import pandas as pd


def project_raw_twitter_to_text_lang(path_to_raw_data):
    """
    Yield text and lang fields from raw tweets
    """
    with open(path_to_raw_data, 'r') as raw:
        for line in raw:
            json_line = json.loads(line)
            yield {
                'text': json_line['data']['text'].replace('\n', '\\n'),
                'lang': json_line['data']['lang']
            }


path_to_raw_twitter_data = './data/raw/twitter_stream.txt'
path_to_processed_data = './data/processed/twitter_text_with_lang.csv'

df = pd.DataFrame(project_raw_twitter_to_text_lang(path_to_raw_twitter_data))
df.to_csv(path_to_processed_data, index=False)
