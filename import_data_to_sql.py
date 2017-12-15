import sqlite3
import pandas as pd
import numpy as np
import os

path_data = "data/per_half_hour"
files = os.listdir(path_data)
csvs = []
for idx, file in enumerate(files):
    lib = pd.read_csv(path_data + "/" + file, sep=";", decimal=",")
    lib["library"] = idx
    csvs += [lib]
libs = pd.concat(csvs, axis=0)
libs["Date"] = pd.to_datetime(libs["Date"], infer_datetime_format=True)


holidays_all = pd.read_csv("data/holidays.csv", sep=";", decimal=",")
cols_holidays = ["Vacances_A", "Vacances_B", "Vacances_C", "Férié"]
holidays = holidays_all.loc[:, ["Date"] + cols_holidays]
holidays.replace(np.nan, -1, inplace=True)
holidays["Date"] = pd.to_datetime(holidays["Date"], infer_datetime_format=True)
for col in cols_holidays:
    holidays[col].astype(np.int8, copy=False)

conn = sqlite3.connect("all.db")
cursor = conn.cursor()

libs.to_sql("libraries", conn, if_exists="replace")
holidays.to_sql("holidays", conn, if_exists="replace")

cursor.execute("drop view if EXISTS data;")
cursor.execute("""
  CREATE VIEW data as
    SELECT
          l."index" as id, library,
          l.Date as date,
          strftime('%s', l.Date) as date_timestamp,
          strftime('%w', l.Date) as day_of_week,
          strftime('%j', l.Date) as day_of_year,
          strftime('%H%M', l.Date) as hour,
          Vacances_A, Vacances_B, Vacances_C,
          Férié, "Visiteurs presents" as visitors
    FROM libraries l JOIN holidays h ON DATE(h.Date) = DATE(l.Date)
    WHERE visitors >= 0
  ;
""")
conn.commit()

data = pd.read_sql_query("""
    select * from data as d 
""", conn, index_col="id")

data.to_csv("data/data_all.gz", sep=";", decimal=",", compression="gzip")
data.to_csv("data/data_all.csv", sep=";", decimal=",")


conn.close()
os.remove("all.db")

