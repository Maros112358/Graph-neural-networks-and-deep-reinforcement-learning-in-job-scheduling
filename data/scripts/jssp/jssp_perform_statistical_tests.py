import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

def test_models(df):
    """Perform Mann-Whitney U rank test on models"""
    grouped_by_model = df.groupby('model')

    test_groups = []
    models = []
    for model_name, model_group in grouped_by_model:
        test_groups.append(model_group['makespan'])
        models.append(model_name)

    results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            _, p = mannwhitneyu(test_groups[i], test_groups[j], alternative='two-sided')
            results.append({'first_model': models[i], 'second_model': models[j], 'p_value': p})

    return results

if __name__ == '__main__':  
    # load data
    df = pd.read_csv('jssp_processsed_data.csv')
    # Group by 'jobs' and 'machines'
    grouped_by_size = df.groupby(['jobs', 'machines'])

    test_results = pd.DataFrame(columns=["jobs", "machines", "number_of_data_points", "p_value"])

    for size_name, size_group in grouped_by_size:
        # make a boxplot
        # plt.figure(figsize=(16, 10))  # Adjusted size
        # plt.title(f'Box Plot of Gaps for {size_name} instances')
        # boxplot_data = [group['gap'] for _, group in size_group.groupby('model')]
        # plt.boxplot(boxplot_data, labels=[model for model in size_group['model'].unique()], vert=False)
        # plt.xlabel('Gap')
        # plt.ylabel('Model')
        # plt.yticks(rotation=45)  # Rotate y-axis labels
        # plot_filename = f'horizontal_boxplot_jobs_{size_name}.png'
        # plt.savefig(plot_filename, bbox_inches='tight')
        # plt.close()
        print(size_name)

        # collect data for each model
        grouped_by_model = size_group.groupby('model')
        test_groups = []
        models = []
        for model_name, model_group in grouped_by_model:
            test_groups.append(model_group['makespan'])
            models.append(model_name)
        
        # Kruskal-Wallis Test
        x = kruskal(*test_groups)
        p_value = x.pvalue
        # test_results.append({
        #     'jobs'
        # })
