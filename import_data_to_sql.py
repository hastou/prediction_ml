import sqlite3
import pandas as pd
import numpy as np
import os

path_data = "data/per_half_hour"
files = os.listdir(path_data)
csvs = []
files = [files[0]]
for idx, file in enumerate(files):
    lib = pd.read_csv(path_data + "/" + file, sep=";", decimal=",")
    lib["library"] = idx
    csvs += [lib]
libs = pd.concat(csvs, axis=0)
libs["Date"] = pd.to_datetime(libs["Date"], format="%d/%m/%Y %H:%M")


holidays_all = pd.read_csv("data/holidays.csv", sep=";", decimal=",")
cols_holidays = ["Vacances_A", "Vacances_B", "Vacances_C", "Férié"]
holidays = holidays_all.loc[:, ["Date"] + cols_holidays]
holidays.replace(np.nan, -1, inplace=True)
holidays["Date"] = pd.to_datetime(holidays["Date"], format="%d/%m/%Y")
for col in cols_holidays:
    holidays[col].astype(np.int8, copy=False)



weather_all = pd.read_csv("data/meteo_all.zip", sep=";", decimal=",", na_values="mq")
cols_weather = ["pmer", "tend", "tend24", "t", "u", "rr24"]
# "tn24", "tx24", "tw"
weather_all = weather_all[weather_all["numer_sta"] == 7149]
weather = weather_all.loc[:, ["formattedDate"] + cols_weather]
weather.dropna(axis=0, inplace=True, how="any")
weather.rename(index=str, columns={"formattedDate": "Date"}, inplace=True)
weather["Date"] = pd.to_datetime(weather["Date"], format="%Y-%m-%d %H:%M:%S")

# print(weather)


# Add in database

conn = sqlite3.connect("all.db")
cursor = conn.cursor()

libs.to_sql("libraries", conn, if_exists="replace")
holidays.to_sql("holidays", conn, if_exists="replace")
weather.to_sql("weather", conn, if_exists="replace")

cursor.execute("drop view if EXISTS data_view;")
cursor.execute("drop view if EXISTS weather_view;")
cursor.execute("drop view if EXISTS data_view_w_weather;")

cursor.execute("""
  CREATE VIEW data_view as
    SELECT
          l."index" as id, library,
          l.Date as Date,
          strftime('%s', l.Date) as date_timestamp,
          strftime('%w', l.Date) as day_of_week,
          strftime('%j', l.Date) as day_of_year,
          strftime('%H%M', l.Date) as hour,
          Vacances_A, Vacances_B, Vacances_C,
          Férié, "Visiteurs presents" as visitors
    FROM libraries l LEFT JOIN holidays h ON DATE(l.Date) = DATE(h.Date)
    WHERE visitors >= 0
""")

cursor.execute("""
  create view weather_view as
      select
          DATE(w.Date) as Date,
          MAX(pmer) as pressure,
          MAX(tend) as pressure_variation_3h,
          MAX(tend24) as pressure_variation,
          MAX(t) as temperature,
          MAX(u) as humidity,
          MAX(rr24) as rainfall
      from weather w group by DATE(w.Date)
""")

cursor.execute("""
  CREATE VIEW data_view_w_weather as
    SELECT 
        d.*, w.*
    FROM data_view d LEFT JOIN weather_view w ON DATE(d.Date) = DATE(w.Date)
    WHERE DATE(d.Date) >= DATE('2015-07-01')
""")

conn.commit()


data = pd.read_sql_query("""
    select * from data_view_w_weather as d 
""", conn, index_col="id")

data.to_csv("data/data_all.gz", sep=";", decimal=",", compression="gzip")
# data.to_csv("data/data_all.csv", sep=";", decimal=",")


conn.close()
os.remove("all.db")

