import pandas as pd 
if __name__ == '__main__':  
    # load data
    df = pd.read_csv('jssp_processsed_data.csv')
    # Group by 'jobs' and 'machines'
    grouped_by_size = df.groupby(['jobs', 'machines'])

    # test_results = pd.DataFrame(columns=["jobs", "machines", "number_of_data_points", "p_value"])

    # for size_name, size_group in grouped_by_size:
    #     print(f"{size_name} {len(size_group)}")
    #     print(size_group['model'])
    #     break
    #     # make a boxplot
    #     # plt.figure(figsize=(16, 10))  # Adjusted size
    #     # plt.title(f'Box Plot of Gaps for {size_name} instances')
    #     # boxplot_data = [group['gap'] for _, group in size_group.groupby('model')]
    #     # plt.boxplot(boxplot_data, labels=[model for model in size_group['model'].unique()], vert=False)
    #     # plt.xlabel('Gap')
    #     # plt.ylabel('Model')
    #     # plt.yticks(rotation=45)  # Rotate y-axis labels
    #     # plot_filename = f'horizontal_boxplot_jobs_{size_name}.png'
    #     # plt.savefig(plot_filename, bbox_inches='tight')
    #     # plt.close()

    #     # collect data for each model
    #     grouped_by_model = size_group.groupby('model')
    #     test_groups = []
    #     models = []
    #     for model_name, model_group in grouped_by_model:
    #         test_groups.append(model_group['makespan'])
    #         models.append(model_name)
        
    #     # Kruskal-Wallis Test
    #     x = kruskal(*test_groups)
    #     print(size_name, x)
    #     insignificant = 0
    #     significant = 0
    #     total_tests = 0

    #     if x.pvalue < 0.05:
    #         print("Performing pairwise comparisons:")
    #         for i in range(len(models)):
    #             for j in range(i + 1, len(models)):
    #                 stat, p = mannwhitneyu(test_groups[i], test_groups[j], alternative='two-sided')
    #                 if p > 0.05:
    #                     insignificant += 1
    #                 else:
    #                     print(f"{size_name} Comparison {models[i]} vs {models[j]}: Stat={stat}, p-value={p}")
    #                     significant += 1
    #                 total_tests += 1
    #     else:
    #         raise Exception('No significant difference found; pairwise comparisons not performed.')

    #     print(f"{total_tests=} {significant=} {insignificant=}")


