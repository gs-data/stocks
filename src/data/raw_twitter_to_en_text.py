import json

def raw_twitter_to_english_text(path_to_raw_data_file):
    """Yield english text from raw tweets

    Parameters
    ----------
    path_to_raw_data_file : str
        path to raw twitter data, relative to the project root
    
    Yields
    ------
    str
        text from tweets
    """
    with open(path_to_raw_data_file, 'r') as raw:
        for line in raw:
            json_line = json.loads(line)
            if json_line['data']['lang'] == 'en':
                yield json_line['data']['text']#.replace('\n', '\\n')