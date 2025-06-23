import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    columns = ['User ID', 'Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Bot Label']
    return df[columns]
