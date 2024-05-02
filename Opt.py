#**************************Import packages******************************
import pyomo
import pyomo.opt
import json
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib as plt

with open('input_data_static.json') as json_file:
    par = json.load(json_file)

   
with open('input_data_dynamic.json') as json_file:
    profile = json.load(json_file)

def _auxDictionary(a):
    temp_dictionary = {}
    if len(a.shape) == 3:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                for dim2 in np.arange(a.shape[2]):
                    temp_dictionary[(dim0+1, dim1+1, dim2+1)] = a[dim0, dim1, dim2]
    elif len(a.shape) == 2:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                temp_dictionary[(dim0+1, dim1+1)] = a[dim0, dim1]
    else:
        for dim0 in np.arange(a.shape[0]):
            temp_dictionary[(dim0+1)] = a[dim0]
    return temp_dictionary
#temp_dict1 = _auxDictionary(loadLimit)

#**************************************Data definition******************************************
data = {}
data['input_dynamic'] = pd.read_csv('input_dynamic.csv')
n_time = data['input_dynamic']['dT'].size


#***************************************Star time definition********************
from datetime import datetime
now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

#***************************************Sets definition********************
model = pyo.ConcreteModel()
model.t = pyo.Set(initialize = np.arange(1, n_time + 1))


#***************************************Parameters definition********************
model.P_PV = pyo.Param(model.t, initialize =_auxDictionary(data['input_dynamic'].to_numpy()[:,3]))
model.P_D = pyo.Param(model.t, initialize =_auxDictionary(data['input_dynamic'].to_numpy()[:,4]))
model.P_PV_max = pyo.Param(initialize = par['P_PV_max'])
model.P_PD_max = pyo.Param(initialize = par['P_D_max'])
model.P_SE_max = pyo.Param(initialize = par['P_SE_max']) 
model.P_ESS_max = pyo.Param(initialize = par['P_ESS_max']) 
model.E_ESS_max = pyo.Param(initialize = par['E_ESS_max'])
model.P_load_max = pyo.Param(initialize = par['P_load_max']) 
model.delta = pyo.Param(model.t, initialize =_auxDictionary(data['input_dynamic'].to_numpy()[:,0]))
model.custo_venda  = pyo.Param(initialize = par['custo_venda'])	# Custo de venda da energia na subestação
model.custo_compra = pyo.Param(initialize = par['custo_compra'])  # Custo de compra da energia na subestação
model.custo_load   = pyo.Param(initialize = par['custo_load'])  	# Custo do gerador distribuído
model.E0             = pyo.Param(initialize = par['E0']) 


model.dT = pyo.Param(model.t, initialize =_auxDictionary(data['input_dynamic'].to_numpy()[:,0]))
model.penalty1 = 1000000
model.penalty2 = 1000000
model.penalty3 = 0.6
model.DegCost = 0.10

#***************************************Variables definition********************
model.P_SE_in =  pyo.Var(model.t, within = pyo.NonNegativeReals, initialize = 0)
model.P_SE_out = pyo.Var(model.t, within = pyo.NonNegativeReals, initialize = 0)
model.P_ESS =    pyo.Var(model.t, within = pyo.Reals)
model.E_ESS =    pyo.Var(model.t, within = pyo.NonNegativeReals, initialize = 0)
model.P_load =   pyo.Var(model.t, within = pyo.NonNegativeReals, initialize = 0)

#***************************************************constraints******************************************************
def _energy_balance(model,t):
    return model.P_SE_in[t] + model.P_PV[t]*model.P_PV_max  == model.P_ESS[t] + model.P_D[t]*model.P_PD_max + model.P_SE_out[t] + model.P_load[t]
model.energy_balance = pyo.Constraint(model.t, rule = _energy_balance)

def _SE_limit(model,t): 
    return model.P_SE_in[t] <= model.P_SE_max
model.SE_limit = pyo.Constraint(model.t, rule = _SE_limit)

def _SE_limit_(model,t): 
    return model.P_SE_out[t] <= model.P_SE_max
model.SE_limit_ = pyo.Constraint(model.t, rule = _SE_limit_)

def _load_limit_(model,t): 
    return model.P_load[t] <= model.P_load_max
model.load_limit_ = pyo.Constraint(model.t, rule = _load_limit_)

def _BESS_balance_(model,t):
        if t == 1:
            return model.E_ESS[t] == model.E0 + model.P_ESS[t]*model.delta[t]
        elif t > 1:
            return model.E_ESS[t] == model.E_ESS[t-1] + model.delta[t]*model.P_ESS[t] 
model.BESS_balance = pyo.Constraint(model.t, rule = _BESS_balance_)


def _BESS_min_(model,t): 
    return -model.P_ESS_max <= model.P_ESS[t]
model.BESS_min = pyo.Constraint(model.t, rule = _BESS_min_)

def _BESS_max_(model,t): 
    return model.P_ESS[t] <= model.P_ESS_max
model.BESS_max = pyo.Constraint(model.t, rule = _BESS_max_)

def _BESS_energy_max_(model,t): 
    return model.E_ESS[t] <= model.E_ESS_max
model.BESS_energy_max = pyo.Constraint(model.t, rule = _BESS_energy_max_)


#************************************************************************Objective Function***********************************************************
def _FOag(model):
    return sum((model.delta[t] * model.custo_compra * model.P_SE_in[t]) - (model.delta[t] * model.custo_venda * model.P_SE_out[t]) + (model.delta[t] * model.custo_load * (model.P_load_max - model.P_load[t])) for t in np.arange(1, n_time + 1)) 
model.FOag = pyo.Objective(rule = _FOag, sense = pyo.minimize)

#************************************************************************Solve the model***********************************************************
from pyomo.opt import SolverFactory
model.write('res_V4_EC.lp',  io_options={'symbolic_solver_labels': True})


opt = pyo.SolverFactory('cplex', executable='C:/Program Files/IBM/ILOG/CPLEX_Studio129/cplex/bin/x64_win64/cplex.exe')
opt.options['LogFile'] = 'res_V4_EC.log'


results = opt.solve(model)#, tee=True)
results.write()

#************************************************************************End Time information***********************************************************
pyo.value(model.FOag)

now = datetime.now()

end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
print("Dif: {}".format(datetime.strptime(end_time, "%H:%M:%S") - datetime.strptime(start_time, "%H:%M:%S")))


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

P_SE_in_df = ext_pyomo_vals(model.P_SE_in)
P_SE_out_df = ext_pyomo_vals(model.P_SE_out)
P_ESS_df = ext_pyomo_vals(model.P_ESS)
E_ESS_df = ext_pyomo_vals(model.E_ESS)
P_load_df = ext_pyomo_vals(model.P_load)
P_load_max_df = ext_pyomo_vals(model.P_load_max)

#@TODO ADDED BY LARISSA:

custo_compra_df = ext_pyomo_vals(model.custo_compra)
custo_venda_df = ext_pyomo_vals(model.custo_venda)
custo_load_df = ext_pyomo_vals(model.custo_load)
Delta_df = ext_pyomo_vals(model.delta)

Import_price = sum([P_SE_in_df[0][t]*Delta_df[0][t]*custo_compra_df[0]
                   for t in np.arange(1, n_time + 1)])

#discharge_cost = sum([PEVdc_df[t][ev]*dT_df[0][t]*cDA_df[0][t]
#                      for ev in np.arange(1, n_evs + 1) for t in np.arange(1, n_time + 1)])

print('Import_price: {}'.format(Import_price))
#print('Discharge cost: {}'.format(discharge_cost))

# FO ACCOUNTS: 
FO_accounts = pd.DataFrame()

# 1:model.delta[t] * model.custo_compra * model.P_SE_in[t]
FO_accounts['delta*custo_compra*P_SE_in'] = Delta_df * custo_compra_df.values[0][0] *  P_SE_in_df #Custo compra = 1

print(custo_venda_df.values[0][0])
# 2: model.delta[t] * model.custo_venda * model.P_SE_out[t]
FO_accounts['delta*custo_venda*P_SE_out'] = Delta_df * custo_venda_df.values[0][0] * P_SE_out_df 

# 3: model.delta[t] * model.custo_load * (model.P_load_max - model.P_load[t])

FO_accounts['delta*custo_load*(P_load_max- P_load)'] = Delta_df * custo_load_df.values[0][0]*(P_load_max_df.values[0][0] - P_load_df)
FO_accounts['final_sum'] = FO_accounts['delta*custo_compra*P_SE_in'] - FO_accounts['delta*custo_venda*P_SE_out'] + FO_accounts['delta*custo_load*(P_load_max- P_load)']
FO_accounts.to_csv("FO_accounts.csv")

import os 
folder = 'RESULTS_' 

if not os.path.exists(folder):
    os.makedirs(folder)
    

P_SE_in_df.to_csv(folder + '/P_SE_in.csv')
P_ESS_df.to_csv(folder + '/P_ESS.csv')
E_ESS_df.to_csv(folder + '/E_ESS.csv')
P_SE_out_df.to_csv(folder + '/P_SE_out.csv')
P_load_df.to_csv(folder + '/P_load.csv')
P_load_max_df.to_csv(folder + '/P_load_max.csv')