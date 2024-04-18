import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('jssp_preprocess_data.csv')

    groups = df.groupby(['jobs', 'machines'])
    for group_name, group in groups:
        print(group_name)