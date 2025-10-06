import pandas as pd
import glob

# path to all csv files
files = glob.glob(".RawLogs/*.csv")

# read and merge all files
df_list = []
for file in files[:7]:
    temp = pd.read_csv(file, sep="\t", encoding="utf-16")
    temp.columns = temp.columns.str.strip()
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# LineLength > 150
filtered = df[df["LineLength"] > 150]

# save
filtered.to_csv("./FilteredLogs.csv", sep=";", index=False, encoding="utf-8")

print("Len: ", len(filtered))
