import pandas as pd
from scipy.stats import kruskal

if __name__ == '__main__':  
    # load data
    df = pd.read_csv('fjsp_preprocess_data.csv')
    df = df[df['gap'].notna()]

    df = df[df['model'].str.endswith('SPT')]
    print(df)

    # Group by 'jobs' and 'machines'
    test_results = pd.DataFrame(columns=["category", "number_of_data_points", "p_value"])

    grouped_by_size = df.groupby(['category'])
    for index, (size_name, size_group) in enumerate(grouped_by_size):
        # collect data for each model
        size_name, size_group = ('all', 0), df
        grouped_by_model = size_group.groupby('model')
        test_groups = []
        models = []
        for model_name, model_group in grouped_by_model:
            test_groups.append(model_group['gap'])
            models.append(model_name)

        # Kruskal-Wallis Test
        _, pvalue = kruskal(*test_groups)
        print(size_name, pvalue)