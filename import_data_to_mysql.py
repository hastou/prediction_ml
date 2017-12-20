from sqlalchemy import create_engine
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
libs.rename(index=str, columns={"Visiteurs presents": "visitors"}, inplace=True)


holidays_all = pd.read_csv("data/holidays.csv", sep=";", decimal=",")
cols_holidays = ["Vacances_A", "Vacances_B", "Vacances_C", "Férié"]
holidays = holidays_all.loc[:, ["Date"] + cols_holidays]
holidays.replace(np.nan, -1, inplace=True)
holidays["Date"] = pd.to_datetime(holidays["Date"], infer_datetime_format=True)
for col in cols_holidays:
    holidays[col].astype(np.int8, copy=False)



weather_all = pd.read_csv("data/meteo_all.zip", sep=";", decimal=",", na_values="mq")
cols_weather = ["pmer", "tend", "tend24", "t", "u", "rr24"]
# "tn24", "tx24", "tw"
weather_all = weather_all[weather_all["numer_sta"] == 7149]
weather = weather_all.loc[:, ["formattedDate"] + cols_weather]
weather.dropna(axis=0, inplace=True, how="any")
weather.rename(index=str, columns={"formattedDate": "Date"}, inplace=True)
weather["Date"] = pd.to_datetime(weather["Date"], infer_datetime_format=True)

# print(weather)


# Add in database
# db = _mysql.connect(host="localhost", user="tbmc", db="tbmc")
engine = create_engine("mysql+mysqldb://tbmc@localhost/tbmc")
db = engine.raw_connection()
cursor = db.cursor()

libs.to_sql("libraries", engine, if_exists="replace", index=False)
holidays.to_sql("holidays", engine, if_exists="replace", index=False)
weather.to_sql("weather", engine, if_exists="replace", index=False)

cursor.execute("drop view if EXISTS data_lib;")
cursor.execute("drop view if EXISTS data_w_weather;")

cursor.execute("""
  CREATE VIEW data_lib as
      SELECT
        library, l.Date as date,
        DATE_FORMAT(l.Date, '%s') as date_timestamp,
        DATE_FORMAT(l.Date, '%w') as day_of_week,
        DATE_FORMAT(l.Date, '%j') as day_of_year,
        DATE_FORMAT(l.Date, '%H%M') as hour,
        Vacances_A, Vacances_B, Vacances_C,
        Férié, visitors
      FROM libraries l LEFT JOIN holidays h ON DATE(l.Date) = DATE(h.Date)
      WHERE visitors >= 0
    ;
""")

cursor.execute("""
  CREATE VIEW data_w_weather as
    SELECT 
        d.*,
        pmer as pressure, tend as pressure_variation_3h,
        tend24 as pressure_variation, t as temperature,
        u as humidity, rr24 as rainfall
    FROM data_lib d LEFT JOIN weather w ON DATE(d.Date) = DATE(w.Date)
    ;
""")
db.commit()

data = pd.read_sql_query("""
    select * from data_w_weather as d 
""", engine, index_col="id")

data.to_csv("data/data_all.gz", sep=";", decimal=",", compression="gzip")
data.to_csv("data/data_all.csv", sep=";", decimal=",")


db.close()
os.remove("all.db")

