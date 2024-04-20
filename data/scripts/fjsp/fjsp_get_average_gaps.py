import pandas as pd

# Custom aggregation function
def mean_std(x):
    return f"{x.mean():.2f} ± {x.std():.2f}"
    
if __name__ == '__main__':
    # load the data
    df = pd.read_csv('fjsp_preprocess_data.csv')

    # remove rows with nan in 'gap'
    df = df[df['gap'].notna()]

    # Create pivot table
    print(df.dtypes)
    pivot_df = df.pivot_table(index='model', columns='category', values='runtime', aggfunc=mean_std, fill_value='-')

    average_gaps_overall = df.groupby('model')['runtime'].agg(mean_std)
    pivot_df['all'] = average_gaps_overall
    print(pivot_df.to_latex())