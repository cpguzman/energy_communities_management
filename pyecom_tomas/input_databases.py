import mysql.connector 
import sys
import pandas as pd

# Open a connection to db and return a cursor.
def db_open():
    try:
        conn = mysql.connector.connect(
            user="ist1100103",
            password="ozgo1085",
            host="db.tecnico.ulisboa.pt",
            database="ist1100103",
            ssl_disabled=True
        )
        cur = conn.cursor()
        print("Connected!")
        return conn, cur
    except mysql.connector.Error as e:
        print(f"Error connecting to DB Platform: {e}")
        sys.exit(1)

# Retrieve all forecast from db, returning a list.
def db_get_all(cur, table):
    query = f"SELECT * FROM {table}"
    try:
        cur.execute(query)
        return cur.fetchall()
    except mysql.connector.Error as err:
        print(f"Error executing query: {err}")

def db_get_all_columns(cur, table):
    query = f"SHOW COLUMNS FROM {table}"
    try:
        cur.execute(query)
        columns = [column[0] for column in cur.fetchall()]
        return columns
    except mysql.connector.Error as err:
        print(f"Error executing query: {err}")

import os

folder = "inputs_database/all_info"

if not os.path.exists(folder):
    os.makedirs(folder)
connection, cursor = db_open()
print(connection, cursor)
generators = db_get_all(cursor, "generators")
generators_columns = db_get_all_columns(cursor, "generators")

df_generators = pd.DataFrame()

for i in range(len(generators_columns)):
    df_generators[generators_columns[i]] = [value[i] for value in generators]

df_generators.to_csv(folder + "/generators.csv")

#----------------------------------------------------------------------------

generators_forecast = db_get_all(cursor, "generators_forecast")
generators_forecast_columns = db_get_all_columns(cursor, "generators_forecast")

df_generators_forecast = pd.DataFrame()

for i in range(len(generators_forecast_columns)):
    df_generators_forecast[generators_forecast_columns[i]] = [value[i] for value in generators_forecast]

df_generators_forecast.to_csv(folder + "/generators_forecast.csv")

#----------------------------------------------------------------------------

loads = db_get_all(cursor, "loads")
loads_columns = db_get_all_columns(cursor, "loads")

df_loads = pd.DataFrame()

for i in range(len(loads_columns)):
    df_loads[loads_columns[i]] = [value[i] for value in loads]

df_loads.to_csv(folder + "/loads.csv")

#----------------------------------------------------------------------------

loads_forecast = db_get_all(cursor, "loads_forecast")
loads_forecast_columns = db_get_all_columns(cursor, "loads_forecast")

df_loads_forecast = pd.DataFrame()

for i in range(len(loads_forecast_columns)):
    df_loads_forecast[loads_forecast_columns[i]] = [value[i] for value in loads_forecast]

df_loads_forecast.to_csv(folder + "/loads_forecast.csv")