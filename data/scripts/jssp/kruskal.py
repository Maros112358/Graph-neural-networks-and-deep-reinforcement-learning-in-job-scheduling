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
    if label == 'wheatley':
        return 'Wheatley'
        
    return label.replace('_', '-')

if __name__ == '__main__':
    df = pd.read_csv('jssp_preprocess_data.csv')
    groups = []
    models = []

    df = df[df['model'].isin(['LRPT', 'ieee_icce_rl_jsp', 'fjsp_drl'])]
    for model_name, model_group in df.groupby('model'):
        groups.append(model_group['gap'])
        models.append(correct_name(model_name))

    # Conducting the Kruskal-Wallis test
    _, kw_p_value = kruskal(*groups)
    print("Kruskal-wallis test:", kw_p_value)

    # Proceed with pairwise Mann-Whitney U tests only if Kruskal-Wallis test is significant
    if kw_p_value < 0.05:
        p_values = []
        comparisons = []
        
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                stat, p = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                comparisons.append(f"{models[i]} vs {models[j]}")
                p_values.append(p)

        
        # Applying the Holm-Bonferroni method to adjust the p-values
        adjusted_p_values = multipletests(p_values, method='holm')
        
        # Printing results
        test_results = pd.DataFrame(columns=["Comparison", "p-value", "Adjusted p-value", "Significant"])
        index = 0
        for comp, p_val, adj_p_val, significant in zip(comparisons, p_values, adjusted_p_values[1], adjusted_p_values[0]):
            test_results.loc[index] = [comp, p_val, adj_p_val, significant]
            index += 1
        print(test_results.to_string(index=False))
        print(len(test_results))
    else:
        print("No significant differences found across groups. No further testing needed.")
