import pandas as pd


if __name__ == '__main__':
    # load the data
    df = pd.read_csv('fjsp_preprocess_data.csv')

    # remove rows with nan in 'gap'
    df = df[df['gap'].notna()]

    print(len(df))