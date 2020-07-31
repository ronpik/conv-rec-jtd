import csv
import json
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Union

import tqdm

from convoscrape.example.text_processing import process_text


EMPTY_RECORD = ["<no text due to error>", "<USER>", "NO_USER"]


def load_tweets_dict(path: str) -> Dict[str, dict]:
    with open(path, 'r', encoding='utf8') as f:
        return {t["id_str"]: t for t in map(json.loads, f) if "id_str" in t}


def iter_data(path: str) -> Iterable[list]:
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        yield from reader


def date2timestamp(date: str) -> int:
    dt = datetime.strptime(date, '%a %b %d %H:%M:%S +0000 %Y')
    return int(datetime.timestamp(dt))


def attach_text_to_record(record: list, tweets_dict: Dict[str, dict]) -> Union[None, list]:
    tweet_id = record[1]
    if tweet_id not in tweets_dict:
        return record + EMPTY_RECORD

    tweet = tweets_dict[tweet_id]
    if "errors" in tweet:
        return record + EMPTY_RECORD

    user_id = tweet["user"]["id_str"]
    raw_text = tweet["text"].replace("\n", " ").replace("\t", " ")
    processed_text = process_text(raw_text)
    timestamp = date2timestamp(tweet["created_at"])
    return record + [raw_text, processed_text, user_id]


if __name__ == "__main__":
    partial_data_path = "US_Election.data"
    tweets_path = "US_Election_tweets.jsonl"
    print("build tweets dict")
    tweets_dict = load_tweets_dict(tweets_path)
    print("done!\niter records:")
    records = iter_data(partial_data_path)
    with open("US_Election-full-text.data", 'w', encoding='utf8') as outf:
        writer = csv.writer(outf, delimiter='\t', lineterminator='\n')
        for record in tqdm.tqdm(records):
            record_with_text = attach_text_to_record(record, tweets_dict)
            if record_with_text[-1] == "NO_USER":
            # if True:
                writer.writerow(record_with_text)


