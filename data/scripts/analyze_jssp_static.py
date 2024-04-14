import pandas as pd
import os
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt

def find_duplicates(df):
    """Counts the number of duplicate rows ignoring seed and runtime"""
    total_duplicates = 0
    grouped = df.groupby(['jobs', 'machines', 'model'])
    for name, group in grouped:
        ignore_columns = ['seed', 'runtime']
        columns_to_check = [col for col in group.columns if col not in ignore_columns]
        duplicates = group[group.duplicated(subset=columns_to_check, keep='first')]
        if not duplicates.empty:
            total_duplicates += len(duplicates)

    return total_duplicates

# Directory where the CSV files are located
directory = './'

# List to hold all dataframes
dataframes = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.startswith("experiment_static_jssp_") and filename.endswith(".csv"):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Rename 'pdr' column to 'pdr' if it exists
        if 'pdr' in df.columns:
            df.rename(columns={'pdr': 'model'}, inplace=True)

        # Remove ".txt" suffix from "instance" column
        if 'instance' in df.columns:
            df['instance'] = df['instance'].str.replace('.txt', '', regex=False)

        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all dataframes into one
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Load the "jssp_details.csv"
details_df = pd.read_csv('jssp_details.csv')

# Join the dataframes on the "instance" column
final_df = pd.merge(concatenated_df, details_df, on='instance', how='inner')
print(len(final_df))
original_number_of_rows = len(final_df)
number_of_duplicates = find_duplicates(final_df)

grouped = final_df.groupby(['jobs', 'machines', 'model'])
updated_dfs = []

for name, group in grouped:
    # print(f"Processing group {name}:")
    # Identify columns to ignore in the duplicate check
    ignore_columns = ['seed', 'runtime']
    columns_to_check = [col for col in group.columns if col not in ignore_columns]
    # Find duplicates
    duplicates = group[group.duplicated(subset=columns_to_check, keep='first')]

    if not duplicates.empty:
        # Calculate the average runtime for each set of duplicate rows
        average_runtime = duplicates['runtime'].mean()
        # Drop all duplicates, keep only the first occurrence
        non_duplicates = group.drop_duplicates(subset=columns_to_check, keep='first')
        # Update the 'runtime' of the first occurrence
        non_duplicates.loc[non_duplicates.index[0], 'runtime'] = average_runtime
        updated_group = non_duplicates
    else:
        updated_group = group  # No duplicates, just use the original group

    updated_dfs.append(updated_group)

# Concatenate all the updated groups back into one DataFrame
final_updated_df = pd.concat(updated_dfs, ignore_index=True)
assert find_duplicates(final_updated_df) == 0
assert original_number_of_rows - number_of_duplicates == len(final_updated_df)

# remove 6x6 instances (not enough data)
final_updated_df = final_updated_df[final_updated_df['jobs'] != 6]
final_updated_df['gap'] = (final_updated_df['makespan'] - final_updated_df['upper_bound']) / (final_updated_df['upper_bound'])

# ======== MAIN ANALYSIS ======

# Group by 'jobs' and 'machines'
grouped_by_size = final_updated_df.groupby(['jobs', 'machines'])

for size_name, size_group in grouped_by_size:
    # make a boxplot
    plt.figure(figsize=(16, 10))  # Adjusted size
    plt.title(f'Box Plot of Gaps for Jobs={size_name[0]}, Machines={size_name[1]}')
    boxplot_data = [group['gap'] for _, group in size_group.groupby('model')]
    plt.boxplot(boxplot_data, labels=[model for model in size_group['model'].unique()], vert=False)
    plt.xlabel('Gap')
    plt.ylabel('Model')
    plt.yticks(rotation=45)  # Rotate y-axis labels
    plot_filename = f'horizontal_boxplot_jobs_{size_name[0]}_machines_{size_name[1]}.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

    # collect data for each model
    grouped_by_model = size_group.groupby('model')
    test_groups = []
    models = []
    for model_name, model_group in grouped_by_model:
        test_groups.append(model_group['gap'])
        models.append(model_name)
    
    # Kruskal-Wallis Test
    x = kruskal(*test_groups)
    print(size_name, x)
    insignificant = 0
    significant = 0
    total_tests = 0
    if x.pvalue < 0.05:
        print("Performing pairwise comparisons:")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                stat, p = mannwhitneyu(test_groups[i], test_groups[j], alternative='two-sided')
                if p > 0.05:
                    insignificant += 1
                else:
                    print(f"Comparison {models[i]} vs {models[j]}: Stat={stat}, p-value={p}")
                    significant += 1
                total_tests += 1
    else:
        print("No significant difference found; pairwise comparisons not performed.")

    print(f"{total_tests=} {significant=} {insignificant=}")


