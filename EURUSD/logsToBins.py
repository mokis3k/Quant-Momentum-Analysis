import pandas as pd

# reading CSV
df = pd.read_csv("./RawLogs/EUR-15M-50.csv", sep="\t", encoding="utf-16")

# cleaning column names
df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=True)

# define ranges (bins) from 50 to 300 in increments of 25.
bins = list(range(25, 325, 25))
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

# Break LineLength down into ranges
df["Range"] = pd.cut(df["LineLength"], bins=bins, labels=labels, right=False)

# calculate the number, average and probability
total = len(df)
result = df.groupby("Range").agg(
    Count=("LineLength", "count"),
    Mean=("LineLength", "mean")
).reset_index()

# round the average and replace NaN with 0.
result["Mean"] = result["Mean"].fillna(0).round(0)

# probability (%)
result["Probability %"] = (result["Count"] / total * 100).round(2)

print(result)

# result.to_csv("line_length_stats.csv", index=False, encoding="utf-8-sig")
