import pandas as pd
from scipy.stats import mannwhitneyu
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
        
    return label.replace('_', '-')

if __name__ == '__main__':  
    df = pd.read_csv('fjsp_preprocess_data.csv')
    df = df[df['gap'].notna()]
    df = df[df['model'].str.endswith('SPT')]

    test_results = pd.DataFrame(columns=["Category", "First PDR", "Second PDR", "p-value"])
    index = 0
    grouped_by_size = df.groupby(['category'])
    models = ['fjsp_drl', "end_to_end_drl_for_fjsp"]
    for size_name, size_group in grouped_by_size:
        results = test_models(size_group, 'model')
        for result in results:
            if result['p_value'] > 0.05:
                test_results.loc[index] = [
                    size_name[0],
                    result['first_model'],
                    result['second_model'],
                    result['p_value']
                ]
                index += 1

    size_name, size_group = ('all', 0), df
    results = test_models(size_group, 'model')
    for result in results:
        if result['p_value'] > 0.05:
            test_results.loc[index] = [
                size_name[0],
                result['first_model'],
                result['second_model'],
                result['p_value']
            ]
            index += 1

    print(test_results.to_latex(index=False))

