import pandas as pd
from scipy.stats import kruskal
import numpy as np

def mean_std(x):
    return f"{x.mean():.2f} ± {x.std():.2f}"

if __name__ == '__main__':  
    # load data
    df = pd.read_csv('djsp_preprocess_data.csv')


    for load_factor in [1, 2, 4]:
        test_results = pd.DataFrame(columns=["Size", "L2D", "IEEE-ICCE-RL-JSP", "Wheatley", "Kruskal test p-value"])
        index = 0
        load_factor_group = df[df['load_factor'] == load_factor]
        for size, size_group in load_factor_group.groupby(['jobs', 'machines']):
            jobs, machines = size
            
            test_groups = []
            models = []
            for model, model_group in size_group.groupby(['model']):
                test_groups.append(model_group['makespan'])
                models.append(model)        

            _, pvalue = kruskal(*test_groups)


            test_results.loc[index] = [
                f"{jobs}x{machines}", 
                mean_std(size_group[size_group['model'] == 'l2d']['makespan']), 
                mean_std(size_group[size_group['model'] == 'ieee_icce_rl_jsp']['makespan']), 
                mean_std(size_group[size_group['model'] == 'wheatley']['makespan']), 
                f"{np.round(pvalue * 100)}$\%$" if pvalue > 0.01 else "< 1$\%$"
            ]
            index += 1

        print(test_results.to_latex(index=False))
