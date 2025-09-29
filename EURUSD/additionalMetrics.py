import pandas as pd

# === settings ===
input_file = "./EURUSD/FilteredLogs.csv"
output_file = "dataset_enriched.csv"

# === loading ===
df = pd.read_csv(input_file, sep=";", encoding="utf-8")
df.columns = df.columns.str.strip()

# === convert dates ===
df["StartTime"] = pd.to_datetime(df["StartTime"], format="%d.%m.%Y %H:%M")
df["EndTime"] = pd.to_datetime(df["EndTime"], format="%d.%m.%Y %H:%M")

# === calculate new columns ===
df["Duration_td"] = df["EndTime"] - df["StartTime"]
df["Duration"] = df["Duration_td"].apply(
    lambda x: f"{int(x.total_seconds()//3600):02d}:{int((x.total_seconds()%3600)//60):02d}"
)

df["Duration_hours"] = df["Duration_td"].dt.total_seconds() / 3600
df["Angle"] = (df["LineLength"] / df["Duration_hours"]).round(3)

# Day of the week
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4
}
df["Weekday"] = df["StartTime"].dt.day_name().map(weekday_map)

# === splitting StartTime and EndTime into dates and timesя ===
df["StartDate"] = df["StartTime"].dt.strftime("%d.%m.%Y")
df["StartTimeOnly"] = df["StartTime"].dt.strftime("%H:%M")
df["EndDate"] = df["EndTime"].dt.strftime("%d.%m.%Y")
df["EndTimeOnly"] = df["EndTime"].dt.strftime("%H:%M")

# === delete temporary and source columns ===
df = df.drop(columns=["Duration_td", "Duration_hours", "StartTime", "EndTime"])

# === reorder columns ===
column_order = [
    "Direction", "StartDate", "StartTimeOnly", "EndDate", "EndTimeOnly",
    "StartPrice", "EndPrice", "LineLength", "Duration", "Angle", "Weekday"
]
df = df[column_order]

# === save ===
df.to_csv(output_file, sep=";", index=False, encoding="utf-8")
print(output_file)
