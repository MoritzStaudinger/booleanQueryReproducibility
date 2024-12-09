import os
import glob
import pandas as pd
from scipy.stats import ttest_1samp

# Directory where your CSV files are located
directory = 'data/3-evaluated/seed/'
query = ("q4")
metric = 'recall'
score_from_paper = 0.129
# Get a list of all CSV files in the directory that end with '_q1.csv'
subdirectories = glob.glob(os.path.join(directory, 'Seed_gpt-3.5-turbo-11*'))
#subdirectories = glob.glob(os.path.join(directory, 'Seed_gpt-3.5-turbo-01*'))
subdirectories = glob.glob(os.path.join(directory, 'Seed_gpt-4*'))

dfs = []

# Iterate over subdirectories
for subdir in subdirectories:
    # Get a list of all CSV files in the subdirectory that end with '_q1.csv'
    csv_files = glob.glob(os.path.join(subdir, f'*_{query}.csv'))

    # Combine all dataframes from CSV files into one dataframe
    combined_df = pd.concat((pd.read_csv(file) for file in csv_files))

    # Append the combined dataframe to the list
    dfs.append(combined_df)

# Combine all dataframes into one dataframe
combined_df = pd.concat(dfs)

# Assuming your dataframe has a column named 'score' which contains the scores
mean_scores_per_run = combined_df[metric]
# The score from the paper


# Perform the t-test
t_stat, p_value = ttest_1samp(mean_scores_per_run, score_from_paper)

# Output the results
print("T-statistic:", t_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    if t_stat < 0:
        print("The mean score is significantly worse than the score from the paper.")
    else:
        print("The mean score is significantly better than the score from the paper.")
else:
    print("There is not enough evidence to conclude whether the mean score is significantly worse or better than the score from the paper.")

