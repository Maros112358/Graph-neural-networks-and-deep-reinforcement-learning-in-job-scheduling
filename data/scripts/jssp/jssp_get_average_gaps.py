import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('jssp_preprocess_data.csv')

    # Custom aggregation function
    def mean_std(x):
        return f"{x.mean():.2f} ± {x.std():.2f}"

    # Create pivot table
    pivot_df = df.pivot_table(index='parameter_set', columns='category', values='gap', aggfunc=mean_std, fill_value='-')

    print(pivot_df.to_latex())