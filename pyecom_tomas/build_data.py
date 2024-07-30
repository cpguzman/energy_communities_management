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

class Data:
    def __init__(self, _file_path='data/EC_V4.xlsx'):
        self.data = HMParser(file_path=_file_path, ec_id=1)
        self.data.parse()
        self.db_config = {
            'host': 'db.tecnico.ulisboa.pt',
            'user': 'ist1100103',
            'password': 'ozgo1085',
            'database': 'ist1100103'
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
            
    def get_gen_data_from_db(self, specific_date, start = 0, end = 24, time_step = 60):

        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()

        select_query = f"SELECT * FROM generators WHERE date = '{specific_date}'"

        cursor.execute(select_query)
        rows = cursor.fetchall()
        cursor.close()
        cursor.close()

        generatorssss = []
        for row in rows:
            json_data_string = row[4].decode('utf-8')
            data_dictionary = json.loads(json_data_string)
            if row[2] != time_step:
                df = pd.DataFrame(list(data_dictionary.items()), columns=['Time', 'Value'])
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df.set_index('Time', inplace=True)
                resampled_df = df.resample('h').mean()
                generatorssss.append([x if x > 0 else 0 for x in resampled_df['Value'].tolist()])
            else:
                generatorssss.append([x if x > 0 else 0 for x in list(data_dictionary.values())])            
        generators = np.array(generatorssss)
        number_of_generators = generators.shape[0]
        
        upac_gen_dict = {}
        for key in self.data.generator.keys():
            if key == 'p_forecast':
                upac_gen_dict[key] = self.data.generator[key][0:number_of_generators, :]
                upac_gen_dict[key][:, start:end] = generators[:, start:end]
            elif key == 'type_generator':
                upac_gen_dict[key] = np.ones((number_of_generators))*2 #All are renewable
            elif len(self.data.generator[key].shape) == 2:
                upac_gen_dict[key] = self.data.generator[key][0:number_of_generators, :]
            elif len(self.data.generator[key].shape) == 1:
                upac_gen_dict[key] = self.data.generator[key][0:number_of_generators]

        self.data.generator = upac_gen_dict
    
    def get_loads_data_from_db(self, specific_date, start = 0, end = 24, time_step = 60):

        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()

        select_query = f"SELECT * FROM loads WHERE date = '{specific_date}'"

        cursor.execute(select_query)
        rows = cursor.fetchall()
        loadssss = []
        for row in rows:
            json_data_string = row[3].decode('utf-8')
            data_dictionary = json.loads(json_data_string)

            if row[1] != time_step:
                df = pd.DataFrame(list(data_dictionary.items()), columns=['Time', 'Value'])
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df.set_index('Time', inplace=True)
                resampled_df = df.resample('h').mean()
                loadssss.append([x if x > 0 else 0 for x in resampled_df['Value'].tolist()])
            else:
                loadssss.append([x if x > 0 else 0 for x in list(data_dictionary.values())])
                
        loadss = np.array(loadssss)
        number_of_laods = loadss.shape[0]

        upac_load_dict = {}
        for key in self.data.load.keys():
            if key == 'p_forecast':
                upac_load_dict[key] = self.data.load[key][0:number_of_laods, :]
                upac_load_dict[key][:, start:end] = loadss[:, start:end]
            elif len(self.data.load[key].shape) == 2:
                upac_load_dict[key] = self.data.load[key][0:number_of_laods, :]
            elif len(self.data.load[key].shape) == 1:
                upac_load_dict[key] = self.data.load[key][0:number_of_laods]
        
        self.data.load = upac_load_dict
    
    def get_gen_forecast_data_from_db(self, specific_date, experiment_id = 13, start = 0, end = 24, time_step = 60):

        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()

        select_query = f"SELECT * FROM generators_forecast WHERE date = '{specific_date}' AND Experiment_ID = {experiment_id}"

        cursor.execute(select_query)
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        
        generatorssss = []
        for row in rows:
            json_data_string = row[4].decode('utf-8')
            data_dictionary = json.loads(json_data_string)
            if row[2] != time_step:
                df = pd.DataFrame(list(data_dictionary.items()), columns=['Time', 'Value'])
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df.set_index('Time', inplace=True)
                resampled_df = df.resample('h').mean()
                generatorssss.append([x if x > 0 else 0 for x in resampled_df['Value'].tolist()])
            else:
                generatorssss.append([x if x > 0 else 0 for x in list(data_dictionary.values())])
            
        generators_forecast = np.array(generatorssss)
        number_of_generators = generators_forecast.shape[0]

        upac_gen_forecast_dict = {}
        for key in self.data.generator.keys():
            if key == 'p_forecast':
                upac_gen_forecast_dict[key] = self.data.generator[key][0:number_of_generators, :]
                upac_gen_forecast_dict[key][:, start:end] = generators_forecast[:, start:end]
            elif key == 'type_generator':
                upac_gen_forecast_dict[key] = np.ones((number_of_generators))*2
            elif len(self.data.generator[key].shape) == 2:
                upac_gen_forecast_dict[key] = self.data.generator[key][0:number_of_generators, :]
            elif len(self.data.generator[key].shape) == 1:
                upac_gen_forecast_dict[key] = self.data.generator[key][0:number_of_generators]


        self.data.generator = upac_gen_forecast_dict
    
    def get_loads_forecast_data_from_db(self, specific_date, experiment_id = 19, start = 0, end = 24, time_step = 60):

        connection = mysql.connector.connect(**self.db_config)
        cursor = connection.cursor()

        select_query = f"SELECT * FROM loads_forecast WHERE date = '{specific_date}' AND Experiment_ID = {experiment_id}"

        cursor.execute(select_query)
        rows = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        loadssss = []
        for row in rows:
            json_data_string = row[3].decode('utf-8')
            data_dictionary = json.loads(json_data_string)
            if row[1] != time_step:
                df = pd.DataFrame(list(data_dictionary.items()), columns=['Time', 'Value'])
                df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
                df.set_index('Time', inplace=True)
                resampled_df = df.resample('h').mean()
                loadssss.append([x if x > 0 else 0 for x in resampled_df['Value'].tolist()])
                
            else:
                loadssss.append([x if x > 0 else 0 for x in list(data_dictionary.values())])
        loadss_forecast = np.array(loadssss)
        number_of_laods = loadss_forecast.shape[0]

        upac_load_forecast_dict = {}
        for key in self.data.load.keys():
            if key == 'p_forecast':
                upac_load_forecast_dict[key] = self.data.load[key][0:number_of_laods, :]
                upac_load_forecast_dict[key][:, start:end] = loadss_forecast[:, start:end]
            elif len(self.data.load[key].shape) == 2:
                upac_load_forecast_dict[key] = self.data.load[key][0:number_of_laods, :]
            elif len(self.data.load[key].shape) == 1:
                upac_load_forecast_dict[key] = self.data.load[key][0:number_of_laods]
        
        self.data.load = upac_load_forecast_dict

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