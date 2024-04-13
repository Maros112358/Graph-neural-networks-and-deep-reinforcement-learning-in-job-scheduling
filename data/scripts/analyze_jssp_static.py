import pandas as pd
import os

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
print(final_updated_df)

