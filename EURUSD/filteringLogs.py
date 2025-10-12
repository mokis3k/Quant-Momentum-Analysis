import pandas as pd
import glob

# path to all csv files
# files = glob.glob("MR-v1/RawLogs/*.csv")
# files = glob.glob("MR-v2/RawLogs/*.csv")
# files = glob.glob("MR-v3/RawLogs/*.csv")
files = glob.glob("MR-v2/MR-v2.1/RawLogs/*.csv")

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
# filtered.to_csv("MR-v1/FilteredLogs.csv", sep=";", index=False, encoding="utf-8")
# filtered.to_csv("MR-v2/FilteredLogs.csv", sep=";", index=False, encoding="utf-8")
# filtered.to_csv("MR-v3/FilteredLogs.csv", sep=";", index=False, encoding="utf-8")
filtered.to_csv("MR-v2/MR-v2.1/FilteredLogs.csv", sep=";", index=False, encoding="utf-8")

print("Len: ", len(filtered))
