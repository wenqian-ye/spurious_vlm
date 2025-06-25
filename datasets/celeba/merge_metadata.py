import pandas as pd
import os


metadata_csv_name = 'metadata.csv'


# Read list_eval_partition.csv with proper column names
df1 = pd.read_csv("list_eval_partition.csv", header=None, sep=r'\s+', names=['filename', 'partition'])

# Read list_attr_celeba.csv, skip first row (count), use second row as header
# Use sep=r'\s+' to handle multiple spaces properly
df2 = pd.read_csv("list_attr_celeba.csv", skiprows=1, sep=r'\s+')

# Reset index to convert filename index to a regular column
df2 = df2.reset_index()
df2 = df2.rename(columns={'index': 'filename'})

merged_df = pd.merge(df1, df2, on='filename')
merged_df["split"] = merged_df["partition"]
merged_df.to_csv(metadata_csv_name, header=True, index=False)