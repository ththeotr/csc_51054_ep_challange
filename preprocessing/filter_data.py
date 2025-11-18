import pandas as pd

def not_na(val):
    if isinstance(val, (list, tuple, set)):
        return not pd.isna(pd.Series(list(val))).all()
    return not pd.isna(val)

def filter_data_v0(filepath, outpath, test=False):
    df = pd.read_json(filepath, lines=True)
    df = pd.json_normalize(df.to_dict(orient='records'))

    with open("preprocessing/columns_v0.txt") as f:
        columns = [l.strip() for l in f.readlines()]
    
    if test:
        df = df[columns[:-1]]
    else:
        df = df[columns]

    def _select(tweet, cname):
        val = tweet[cname]
        if cname == "text":
            cname = "full_text"
        if not_na(tweet[f'extended_tweet.{cname}']):
            val = tweet[f'extended_tweet.{cname}']
        return val
    for col in columns[3:6]:
        df[col] = df.apply(lambda x: _select(x, col), axis=1)
    
    df = df.drop(columns[:3], axis=1)

    for col in columns[4:6]:
        df[col] = df[col].apply(lambda x: len(x))

    def _replace_nan(tweet):
        if pd.isna(tweet["user.description"]):
            tweet["user.description"] = ""
        return tweet

    df = df.apply(lambda x: _replace_nan(x), axis=1)

    df.to_json(outpath, orient='records', lines=True, date_format='iso', force_ascii=False)


def filter_data_v1(filepath, outpath, test=False):
    df = pd.read_json(filepath, lines=True)
    df = pd.json_normalize(df.to_dict(orient='records'))

    with open("preprocessing/columns_v1.txt") as f:
        columns = [l.strip() for l in f.readlines()]
    
    if test:
        df = df[columns[:-1]]
    else:
        df = df[columns]

    df.to_json(outpath, orient='records', lines=True, date_format='iso', force_ascii=False)


if __name__ == "__main__":
    filter_data_v0(
        "Kaggle2025/train.jsonl",
        "data/train_clean_v0.jsonl",
        False
    )
    filter_data_v0(
        "Kaggle2025/kaggle_test.jsonl",
        "data/test_clean_v0.jsonl",
        True
    )
    filter_data_v1(
        "data/train_clean_v0.jsonl",
        "data/train_clean_v1.jsonl",
        False
    )
    filter_data_v1(
        "data/test_clean_v0.jsonl",
        "data/test_clean_v1.jsonl",
        True
    )
