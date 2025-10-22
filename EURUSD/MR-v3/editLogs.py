import pandas as pd

input_file = "logsMomentumLines.csv"
output_file = "dataset.csv"

# read the file
df = pd.read_csv(input_file, sep="\t", encoding="utf-16")

# delete the first five columns
df = df.drop(df.columns[0:5], axis=1)

# required columns
columns_to_keep = [
    "LineLength",
    "AvgDiff",
    "LastCandlesDiff",
    "PrevLineLength",
    "MaxDeviation",
    "DurationHours",
    "BarsCount"
]
df = df[columns_to_keep]

# delete the first row (initial line)
df = df.iloc[1:].reset_index(drop=True)

df.to_csv(output_file, index=False)
print("File saved:", output_file)
