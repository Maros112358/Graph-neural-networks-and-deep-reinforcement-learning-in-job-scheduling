import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('jssp_preprocess_data.csv')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

    # Custom aggregation function
    def mean_std(x):
        return f"{x.mean():.2f} ± {x.std():.2f}"

    # Create pivot table
    print(df.dtypes)
    pivot_df = df.pivot_table(index='model', columns='category', values='runtime', aggfunc=mean_std, fill_value='-')

    average_gaps_overall = df.groupby('model')['runtime'].agg(mean_std)
    pivot_df['all'] = average_gaps_overall
    # print(pivot_df)
    print(pivot_df.to_latex())

# if __name__ == "__main__":
