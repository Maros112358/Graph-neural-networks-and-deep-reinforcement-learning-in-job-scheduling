import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import numpy as np

def mean_std(x):
    return f"{x.mean():.2f} ± {x.std():.2f}"

def correct_label(label: str) -> str:
    if label == 'end_to_end_drl_for_fjsp':
        return 'End-to-end-DRL-for-FJSP'
    if label == 'fjsp_drl':
        return 'fjsp-drl'
    if label == 'ieee_icce_rl_jsp':
        return 'IEEE-ICCE-RL-JSP'
    if label == 'wheatley':
        return 'Wheatley'
    if label == 'l2d':
        return 'L2D'
        
    return label.replace('_', '-')

if __name__ == '__main__':  
    # load data
    df = pd.read_csv('djsp_preprocess_data.csv')


    test_results = pd.DataFrame(columns=["Size", "Load factor", "First model", "Second model", "p-value"])
    index = 0
    for size, size_group in df.groupby(['jobs', 'machines']):
        jobs, machines = size
        for load_factor in [1, 2, 4]:
            load_factor_group = size_group[size_group['load_factor'] == load_factor]
            
            test_groups = []
            models = []
            for model, model_group in load_factor_group.groupby(['model']):
                test_groups.append(model_group['makespan'])
                models.append(model)        

            _, pvalue = kruskal(*test_groups)

            
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    _, pvalue = mannwhitneyu(test_groups[i], test_groups[j], alternative='two-sided')
                    if pvalue > 0.05:
                        test_results.loc[index] = [
                            f"{jobs}x{machines}",
                            load_factor,
                            correct_label(models[i][0]),
                            correct_label(models[j][0]),
                            f"{np.round(pvalue * 100)}$\%$"
                        ]
                        index += 1

    print(test_results.to_latex(index=False))
