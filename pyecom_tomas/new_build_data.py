# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import pyomo
import pyomo.opt
import pyomo.environ as pe
import json
import mysql.connector
from src.parsers import HMParser
import os

class Data:
    def __init__(self, _file_path='data/EC_V4_new_UC2.xlsx'):
        self.data = HMParser(file_path=_file_path, ec_id=1)
        self.data.parse()
        self.db_config = {
            'host': 'db.tecnico.ulisboa.pt',
            'user': 'ist1100103',
            'password': 'ozgo1085',
            'database': 'ist1100103',
            'ssl_disabled': True
        }
    
    def get_data(self):
        return self.data
        
    def increase_number_of_storage(self, factor):
        for key in self.data.storage.keys():
            if len(self.data.storage[key].shape) == 1:
                self.data.storage[key] = np.tile(self.data.storage[key], (factor, ))
            else:
                # Add in n times
                self.data.storage[key] = np.tile(self.data.storage[key], (factor, 1))
                
    def increase_number_of_vehicle(self, factor):
        for key in self.data.vehicle.keys():
            if len(self.data.vehicle[key].shape) == 1:
                self.data.vehicle[key] = np.tile(self.data.vehicle[key], (factor, ))
            elif len(self.data.vehicle[key].shape) == 2:
                self.data.vehicle[key] = np.tile(self.data.vehicle[key], (factor, 1))
            elif len(self.data.vehicle[key].shape) == 3:
                factor_sc = self.data.charging_station['p_discharge_limit'].shape[0]//3
                self.data.vehicle[key] = np.tile(self.data.vehicle[key], (factor_sc, factor, 1))
    
    def increase_number_of_charging_station(self, factor):
        for key in self.data.charging_station.keys():
            shape = self.data.charging_station[key].shape
            if len(shape) == 1:
                self.data.charging_station[key] = np.tile(self.data.charging_station[key], (factor,))
            elif len(shape) == 2:
                self.data.charging_station[key] = np.tile(self.data.charging_station[key], (factor, 1))
            elif len(shape) == 3:
                self.data.charging_station[key] = np.tile(self.data.charging_station[key], (factor, 1, 1))
            else:
                raise ValueError(f"Unsupported array shape: {shape}")
            
    def get_data_from_db(self, specific_date, table, start = 0, end = 24, time_step = 60, 
                         save=False, experiment_id=None, folder=None):

        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()
        
        # Getting Information from DB:
        select_query = f"SELECT * FROM {table} WHERE date = '{specific_date}'"

        if experiment_id:
            select_query = f"SELECT * FROM {table} WHERE date = '{specific_date}' AND Experiment_ID = {experiment_id}"

        cursor.execute(select_query)
        rows = cursor.fetchall()

        # Getting Columns name:
        select_columns = f"SELECT * FROM {table} LIMIT 0"
        
        cursor.execute(select_columns)
        column_info = cursor.description
        # Extract the names of the columns
        column_names = [column[0] for column in column_info]
        
        cursor.fetchall()

        cursor.close()
        connection.close()
        
        db_info = []
        self.upacs = []
        
        for row in rows:
            self.upacs.append(row[column_names.index("upac")])
            json_data_string = row[column_names.index("time_series")].decode('utf-8')
            data_dictionary = json.loads(json_data_string)

            # Since all data from mysql databases come in W, and in Excel is in kW, we divide by 1000
            for key in data_dictionary:
                data_dictionary[key] = data_dictionary[key] / 1000 # It becomes in kW.

            if row[2] != time_step:
                df = pd.DataFrame(list(data_dictionary.items()), columns=['Time', 'Value'])
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df.set_index('Time', inplace=True)
                resampled_df = df.resample('h').mean()
                db_info.append([x if x > 0 else 0 for x in resampled_df['Value'].tolist()])
            else:
                db_info.append([x if x > 0 else 0 for x in list(data_dictionary.values())])            
        
        db_info = np.array(db_info)
        quantity_of_db_info = db_info.shape[0]
        
        upac_dict = {}
        if table == "generators" or table == "generators_forecast":
            for key in self.data.generator.keys():
                if key == 'p_forecast':
                    upac_dict[key] = self.data.generator[key][0:quantity_of_db_info, :]
                    upac_dict[key][:, start:end] = db_info[:, start:end]
                elif key == 'type_generator':
                    upac_dict[key] = np.ones((quantity_of_db_info))*2 #All are renewable
                elif len(self.data.generator[key].shape) == 2:
                    upac_dict[key] = self.data.generator[key][0:quantity_of_db_info, :]
                elif len(self.data.generator[key].shape) == 1:
                    upac_dict[key] = self.data.generator[key][0:quantity_of_db_info]
            self.data.generator = upac_dict
        
        elif table == "loads" or table == "loads_forecast":
            for key in self.data.load.keys():
                if key == 'p_forecast':
                    upac_dict[key] = self.data.load[key][0:quantity_of_db_info, :]
                    upac_dict[key][:, start:end] = db_info[:, start:end]
                elif len(self.data.load[key].shape) == 2:
                    upac_dict[key] = self.data.load[key][0:quantity_of_db_info, :]
                elif len(self.data.load[key].shape) == 1:
                    upac_dict[key] = self.data.load[key][0:quantity_of_db_info]
            self.data.load = upac_dict

        if save:
            self.save_data_from_db(table, upac_dict, start, end, folder)

        
    def save_data_from_db(self, archive_name, dict_, start, end, folder=None, excel_=True, json_=True):
        
        dict_info = {}
        for key_ in dict_.keys():
            my_list = dict_[key_].tolist()
            dict_info[key_] = {}
            for i in range(len(self.upacs)):
                dict_info[key_][self.upacs[i]] = my_list[i]

        if folder == None:
            folder = f"./inputs_database/"
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if json_:
            with open(folder + f"/{archive_name}{start}-{end}.json", 'w') as json_file:
                json.dump(dict_info, json_file, indent=4)
        
        if excel_:
            # Create a dictionary to store the DataFrames
            dfs_ = {key: pd.DataFrame(value).T for key, value in dict_.items()}

            # Rename columns
            for df in dfs_.values():
                #df.columns = [f'Upac {upacs[i]}' for i in range(df.shape[1])]
                df.columns = [f'House {i + 1}' for i in range(df.shape[1])]

            # Write the DataFrames to an Excel file, each in a different sheet
            with pd.ExcelWriter(folder + f"/{archive_name}_data{start}-{end}.xlsx") as writer:
                for key, df in dfs_.items():
                    df.to_excel(writer, sheet_name=key, index=False)

    
    def increase_p_forecast(self, input_data, factor):
        input_data["p_forecast"] *= factor
        
    def change_initial_state_storage(self, init_state):
        self.data.storage['initial_state'] = init_state

    def increase_storage_capacity(self, factor):
        self.data.storage['p_charge_limit'] *= factor
        self.data.storage['p_discharge_limit'] *= factor
        self.data.storage['energy_capacity'] *= factor
        self.data.storage['p_charge_max'] *= factor
        self.data.storage['p_discharge_max'] *= factor
        
    def increase_import_limit(self, factor):
        self.data.peers['import_contracted_p_max'][0, :]*=factor
        
    def print_data_attributes(self):
        print("Model has the following attributes:")
        for key, value in vars(self.data).items():
            if not key.startswith('__'):
                print(key)

    def update_forecast_integral(self, integral_of_error_gen, integral_of_error_load, i):
        new1 = np.maximum(self.data.generator['p_forecast'][:, i:24] + integral_of_error_gen , 0)
        new2 = np.maximum(self.data.load['p_forecast'][:, i:24] + integral_of_error_load , 0)
        self.data.generator['p_forecast'][:, i:24] = new1
        self.data.load['p_forecast'][:, i:24] = new2
        
    def update_forecast_low_pass(self, error_load, error_gen, gamma, i):
        gen = np.zeros((5, 1))
        load = np.zeros((5, 1))
        for j in range(i):
            gen = gen + gamma**(j-i)*error_gen[:, j]
            load = load + gamma**(j-i)*error_load[:, j]
            print("AIAI: ", gen)
            
        new1 = np.maximum(self.data.generator['p_forecast'][:, i:24] + gen , 0)
        new2 = np.maximum(self.data.load['p_forecast'][:, i:24] + load , 0)
        self.data.generator['p_forecast'][:, i:24] = new1
        self.data.load['p_forecast'][:, i:24] = new2
        
    def change_time_step(self, time_step):
        div = 60//time_step
        for key, value in vars(self.data).items():
            dic_aux = {}
            if type(value) == dict:
                for key2, value2 in value.items():
                    if len(value2.shape) != 1:
                        dic_aux[key2] = np.repeat(value2, div, axis=len(value2.shape)-1)
                    else:
                        dic_aux[key2] = value2
            setattr(self.data, key, dic_aux)
            
    def creat_new_cs_schedual(self):
        time_interval = self.data.vehicle['schedule_cs_usage'].shape[2]
        n_cs = self.data.vehicle['schedule_cs_usage'].shape[0]
        n_ve = self.data.vehicle['schedule_cs_usage'].shape[1]
        
        new_schedual = np.zeros((n_cs, n_ve, time_interval))
        asg_ve = 0
        
        while asg_ve < n_ve:
            for event in range(self.data.vehicle['departure_time_period'].shape[1]):
                arv = int(self.data.vehicle['arrive_time_period'][asg_ve, event]) - 1
                dep = int(self.data.vehicle['departure_time_period'][asg_ve, event])
                not_enough = True
                for c in range(n_cs):
                    if new_schedual[c, :, arv:dep].sum() == 0:
                        new_schedual[c, asg_ve, arv:dep] = 1
                        
                        not_enough = False
                        break
                    
                if not_enough == True:
                    print("Problem with car ", asg_ve)
                    print("Could not assign all vehicles to charging stations")
                    self.data.vehicle['schedule_cs_usage']= new_schedual
                    return None
                
            asg_ve += 1
                
        self.data.vehicle['schedule_cs_usage'] = new_schedual
        return new_schedual
        
# Auxiliary function to convert numpy arrays to dictionaries
def convert_to_dictionary(a, t_start=0):
    temp_dictionary = {}

    if len(a.shape) == 3:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                for dim2 in np.arange(a.shape[2]):
                    temp_dictionary[(dim0+1, dim1+1, dim2+1+t_start)] = a[dim0, dim1, dim2]
    elif len(a.shape) == 2:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                temp_dictionary[(dim0+1, dim1+1+t_start)] = a[dim0, dim1]

    else:
        for dim0 in np.arange(a.shape[0]):
            temp_dictionary[(dim0+1+t_start)] = a[dim0]

    return temp_dictionary

# Extract Pyomo values
# https://stackoverflow.com/questions/67491499/how-to-extract-indexed-variable-information-in-pyomo-model-and-build-pandas-data
def ext_pyomo_vals(vals):
    # make a pd.Series from each
    s = pd.Series(vals.extract_values(),
                  index=vals.extract_values().keys())

    # if the series is multi-indexed we need to unstack it...
    if type(s.index[0]) == tuple:    # it is multi-indexed
        s = s.unstack(level=1)
    else:
        # force transition from Series -> df
        s = pd.DataFrame(s)

    return s