import pandas as pd

# === settings ===
input_file = "./FilteredLogs.csv"
output_file = "./EnrichedDataset.csv"

# === loading ===
df = pd.read_csv(input_file, sep=";", encoding="utf-8")
df.columns = df.columns.str.strip()

# === convert dates ===
df["StartTime"] = pd.to_datetime(df["StartTime"], format="%d.%m.%Y %H:%M")
df["EndTime"] = pd.to_datetime(df["EndTime"], format="%d.%m.%Y %H:%M")

# === calculate Duration and Angle ===
df["Duration_td"] = df["EndTime"] - df["StartTime"]
df["Duration"] = df["Duration_td"].dt.total_seconds() / 60  # минуты
df["Duration_hours"] = df["Duration_td"].dt.total_seconds() / 3600
df["Angle"] = (df["LineLength"] / df["Duration_hours"]).round(5)

# === determine the trading session ===
def get_session(t):
    hour = t.hour
    if 0 <= hour < 9:
        return 0  # AS
    elif 9 <= hour < 17:
        return 1  # LO
    else:
        return 2  # NY

df["Session"] = df["StartTime"].apply(get_session)

# === format StartTime and remove unnecessary columns ===
df["StartTime"] = df["StartTime"].dt.strftime("%H:%M:%S")
df = df.drop(columns=["EndTime", "Duration_td", "Duration_hours"])

# === save ===
df.to_csv(output_file, sep=";", index=False, encoding="utf-8")

print("File saved", output_file)
