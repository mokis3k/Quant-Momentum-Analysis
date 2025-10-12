import pandas as pd
import glob

files = glob.glob("RawLogs/*.csv")

# read and merge all files
df_list = []
for file in files[:7]:
    temp = pd.read_csv(file, sep="\t", encoding="utf-16")
    temp.columns = temp.columns.str.strip()
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
df.to_csv("GroupedLogs.csv", sep=";", index=False, encoding="utf-8")

print("Len: ", len(df))
