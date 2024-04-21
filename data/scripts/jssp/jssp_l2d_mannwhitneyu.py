import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt

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
    if label == 'ieee_icce_rl_jsp':
        return 'IEEE-ICCE-RL-JSP'
    if label == 'LRPT':
        return 'MWKR'
        
    return label.replace('_', '-')

if __name__ == '__main__':  
    df = pd.read_csv('jssp_preprocess_data.csv')

    test_results = pd.DataFrame(columns=["Category", "First model", "Second model", "p-value"])
    index = 0

    models = ['l2d', 'l2d_large']
    df = df[df['model'].isin(models)]

    for size_name, size_group in df.groupby(['category']):
        grouped_data = size_group.groupby('model')
        boxplot_data = [group['gap'] for model, group in grouped_data]
        means = [group['gap'].mean() for model, group in grouped_data]
        labels = [model for model in size_group['model'].unique()]

        # make a boxplot
        plt.figure(figsize=(16, 10))  # Adjusted size
        boxplot_data = [group['gap'] * 100 for _, group in size_group.groupby('model')]
        box = plt.boxplot(boxplot_data, labels=[correct_label(model) for model in size_group['model'].unique()], patch_artist=True)
        # Highlight the boxplot with the lowest mean
        for patch in box['boxes']:
            patch.set_facecolor('lightgray')  # Default color
        plt.ylabel('Gap [%]', fontsize=22)
        plt.xticks(rotation=45, fontsize=18, ha='right', rotation_mode='anchor')  # Rotate y-axis labels
        max_gap = max([max(group) for group in boxplot_data])
        max_y_limit = min(1, max_gap) * 100 # Cap the y-axis at 1 or lower if the max value is less than 1
        plt.gca().set_ylim(top=max_y_limit, bottom=-5) 
        # plot_filename = f'/home/maros_b/Graph-neural-networks-and-deep-reinforcement-learning-in-job-scheduling/thesis/images/horizontal_boxplot_jssp_{size_name[0]}.pdf'
        plot_filename = f'horizontal_boxplot_jssp_{size_name[0]}.pdf'
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()


        results = test_models(size_group, 'model')
        print(len(size_group), size_name)
        for result in results:
            if not result['first_model'] in models:
                continue

            if result['second_model'] not in models:
                continue


            # print(size_name, result)
            test_results.loc[index] = [size_name[0],
             f"{result['first_model']}",
             f"{result['second_model']}",
             result['p_value']
            ]
            index += 1

    size_group = df
    size_name = ('all', )
    results = test_models(size_group, 'model')
    print(len(size_group), size_name)
    for result in results:
        if not result['first_model'] in models:
            continue

        if result['second_model'] not in models:
            continue


        test_results.loc[index] = [size_name[0], result['first_model'], result['second_model'], result['p_value']]
        index += 1

    print(test_results.to_latex(index=False))

    # test_results.to_csv('jssp_mannwhitneyu_test.csv', index=False)
