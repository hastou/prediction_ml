import sqlite3
import pandas as pd
import numpy as np



library = pd.read_csv("data/per_half_hour/biblio_1_per_half_hour.csv", sep=";", decimal=",")
library["Date"] = pd.to_datetime(library["Date"], infer_datetime_format=True)
library["library"] = 1

holidays_all = pd.read_csv("data/holidays.csv", sep=";", decimal=",")
cols_holidays = ["Vacances_A", "Vacances_B", "Vacances_C", "Férié"]
holidays = holidays_all.loc[:, ["Date"] + cols_holidays]
holidays.replace(np.nan, 0, inplace=True)
holidays["Date"] = pd.to_datetime(holidays["Date"], infer_datetime_format=True)

for col in cols_holidays:
    holidays[col] = holidays[col].astype("bool")

conn = sqlite3.connect("all.db")
cursor = conn.cursor()

library.to_sql("libraries", conn, if_exists="replace")
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
          Vacances_A, Vacances_B, Vacances_C,
          Férié, "Visiteurs presents" as visitors
    FROM libraries l JOIN holidays h ON DATE(h.Date) = DATE(l.Date)
    WHERE "Visiteurs presents" > 0
  ;
""")
conn.commit()


