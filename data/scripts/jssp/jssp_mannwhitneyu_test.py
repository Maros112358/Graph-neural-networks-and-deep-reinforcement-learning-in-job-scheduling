import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == '__main__':  
    df = pd.read_csv('jssp_preprocess_data.csv')

    test_results = pd.DataFrame(columns=["category", "first_model", "second_model", "p_value"])

    index = 0
    grouped_by_size = df.groupby(['category'])
    for size_name, size_group in grouped_by_size:
        # Grouping data and calculating means
        grouped_data = size_group.groupby('model')
        boxplot_data = [group['gap'] for _, group in grouped_data]
        means = [group['gap'].mean() for _, group in grouped_data]
        labels = [model for model in size_group['model'].unique()]

        # Find the index of the boxplot with the lowest mean gap
        min_mean_index = means.index(min(means))

        # make a boxplot
        # plt.figure(figsize=(16, 10))  # Adjusted size
        # plt.title(f'Box Plot of Gaps for {size_name[0]} instances')
        # boxplot_data = [group['gap'] for _, group in size_group.groupby('model')]
        # box = plt.boxplot(boxplot_data, labels=[model for model in size_group['model'].unique()], patch_artist=True)
        # # Highlight the boxplot with the lowest mean
        # for patch in box['boxes']:
        #     patch.set_facecolor('lightgray')  # Default color
        # box['boxes'][min_mean_index].set_facecolor('cyan')  # Highlight color
        # plt.xlabel('Gap')
        # plt.ylabel('Model')
        # plt.xticks(rotation=45)  # Rotate y-axis labels
        # max_gap = max([max(group) for group in boxplot_data])
        # max_y_limit = min(1, max_gap)  # Cap the y-axis at 1 or lower if the max value is less than 1
        # plt.gca().set_ylim(top=max_y_limit) 
        # plot_filename = f'horizontal_boxplot_jssp_{size_name[0]}.png'
        # plt.savefig(plot_filename, bbox_inches='tight')
        # plt.close()

        results = test_models(size_group, 'parameter_set')
        for result in results:
            if result['first_model'].startswith('DQN') or result['second_model'].startswith("DQN"):
                if result['p_value'] > 0.05:
                    print(size_name, result)

            if result['first_model'].startswith('DQN') and result['second_model'].startswith("DQN"):
                if result['p_value'] < 0.05:
                    print("DIFFERENT", size_name, result)
                    # raise Exception('wtf')
            test_results.loc[index] = [size_name[0], result['first_model'], result['second_model'], result['p_value']]
            index += 1

    # test_results.to_csv('jssp_mannwhitneyu_test.csv', index=False)
