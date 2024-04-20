import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt

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

def test_models(df, column='model'):
    """Perform Mann-Whitney U rank test on models"""
    grouped_by_model = df.groupby(column)

    test_groups = []
    models = []
    for model_name, model_group in grouped_by_model:
        test_groups.append(model_group['gap'])
        models.append(model_name)

    results = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            _, p = mannwhitneyu(test_groups[i], test_groups[j], alternative='two-sided')
            results.append({'first_model': models[i], 'second_model': models[j], 'p_value': p})

    return results

def correct_label(label: str) -> str:
    if label == 'end_to_end_drl_for_fjsp':
        return 'End-to-end-DRL-for-FJSP'
    if label == 'fjsp_drl':
        return 'fjsp-drl'
        
    return label.replace('_', '-')

if __name__ == '__main__':  
    df = pd.read_csv('fjsp_preprocess_data.csv')
    df = df[df['gap'].notna()]

    test_results = pd.DataFrame(columns=["Category", "Models pair", "p-value"])

    index = 0
    grouped_by_size = df.groupby(['category'])
    for size_name, size_group in grouped_by_size:
        size_name, size_group = ('all', 0), df
        # Grouping data and calculating means
        grouped_data = size_group.groupby('model')
        boxplot_data = [group['gap'] for model, group in grouped_data ]
        means = [group['gap'].mean() for model, group in grouped_data ]
        labels = [model for model in size_group['model'].unique()]

        # Find the index of the boxplot with the lowest mean gap
        min_mean_index = means.index(min(means))
        print(min_mean_index)

            # make a boxplot
        plt.figure(figsize=(16, 10))  # Adjusted size
        # plt.title(f'JSSP gaps for "{size_name[0]}" instances', fontsize=20)
        boxplot_data = [group['gap'] * 100 for _, group in size_group.groupby('model')]
        print(size_name)
        print([len(data) for data in boxplot_data])
        box = plt.boxplot(boxplot_data, labels=[correct_label(model) for model in size_group['model'].unique()], patch_artist=True)
        # Highlight the boxplot with the lowest mean
        for patch in box['boxes']:
            patch.set_facecolor('lightgray')  # Default color
        box['boxes'][min_mean_index].set_facecolor('cyan')  # Highlight color
        plt.ylabel('Gap [%]', fontsize=22)
        # plt.xlabel('Model', fontsize=18)
        plt.xticks(rotation=45, fontsize=18, ha='right', rotation_mode='anchor')  # Rotate y-axis labels
        max_gap = max([max(group) for group in boxplot_data])
        max_y_limit = min(1, max_gap) * 100 # Cap the y-axis at 1 or lower if the max value is less than 1
        # plt.gca().set_ylim(top=1000, bottom=-5) 
        plot_filename = f'/home/maros_b/Graph-neural-networks-and-deep-reinforcement-learning-in-job-scheduling/thesis/images/horizontal_boxplot_fjsp_{size_name[0]}.pdf'
        print(plot_filename)
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()
