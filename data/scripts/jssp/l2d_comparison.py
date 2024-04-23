import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.multitest import multipletests

def correct_name(label: str) -> str:
    if label == 'end_to_end_drl_for_fjsp':
        return 'End-to-end-DRL-for-FJSP'
    if label == 'fjsp_drl':
        return 'fjsp-drl'
    if label == 'ieee_icce_rl_jsp':
        return 'IEEE-ICCE-RL-JSP'
    if label == 'LRPT':
        return 'MWKR'
    if label == 'l2d':
        return "L2D"
    if label == 'l2d_large':
        return 'L2D-LARGE'
    if label == 'wheatley':
        return 'Wheatley'
        
    return label.replace('_', '-')

if __name__ == '__main__':
    df = pd.read_csv('jssp_preprocess_data.csv')
    df = df[df['model'].isin(['l2d', 'l2d_large'])]

    grouped_by_category = [(size_name, size_group) for size_name, size_group in df.groupby('category')] + [('all', df)]
    p_values = []
    categories = []
    for size_name, size_group in grouped_by_category:
        groups = []
        models = []
        categories.append(size_name)
        for model_name, model_group in size_group.groupby('model'):
            groups.append(model_group['gap'])
            models.append(correct_name(model_name))
        
        _, kw_p_value = kruskal(*groups)
        p_values.append(kw_p_value)

    print(p_values)
    adjusted_p_values = multipletests(p_values, method='holm')


    p_values = []
    comparisons = []
    test_results = pd.DataFrame(columns=["Category", "Adjusted p-value"])
    index = 0
    for category, adjusted_p_value, significant, (size_name, size_group) in zip(categories, adjusted_p_values[1], adjusted_p_values[0], grouped_by_category):
        test_results.loc[index] = [category, str(np.round(100*adjusted_p_value))]
        index += 1
        if not significant:
            print(f"Category {category} is not significant, adjusted p-value: {adjusted_p_value}")
            continue
        
        models = []
        groups = []
        for model_name, model_group in size_group.groupby('model'):
            groups.append(model_group['gap'])
            models.append(correct_name(model_name))

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                stat, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                comparisons.append((size_name, models[i], models[j]))
                p_values.append(p)

    print(test_results.to_latex(index=False))

    if p_values:
        adjusted_p_values = multipletests(p_values, method='holm')
        test_results = pd.DataFrame(columns=["Category", "Comparison", "Adjusted p-value"])
        index = 0
        for (category, first_model, second_model), significant, adjusted_p_value in zip(comparisons, adjusted_p_values[0], adjusted_p_values[1]):
            comparison = f'{first_model} vs. {second_model}'
            print(significant)
            test_results.loc[index] = [category.capitalize(), comparison, str(np.round(100*adjusted_p_value,1))]
            index += 1

        print(test_results.to_latex(index=False))
            
