import numpy as np
import pyomo.environ as pe
from build_data import convert_to_dictionary
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd

#Functions to the model
# Set default behaviour
default_behaviour = pe.Constraint.Skip


project_colors = {"green": "#58c1ae", "blue": "#0e73b9", "lightgreen": "#cdd451",
                  "babyblue": "#5ebcea", "orange": "#e58033", "yellow": "#eacf5e",
                  "raw": "#ece9d6"}
project_colors_list = ["#58c1ae", "#0e73b9", "#cdd451", "#5ebcea", "#e58033","#eacf5e","#ece9d6"] 

# Limits for the system, grid import power
def _impMaxEq(m, t):
    return m.imports[t] <= m.impMax[t]

# Limits for the system, grid import power
def _expMaxEq(m, t):
    return m.exports[t] <= m.expMax[t]

# Upper limit for the PV generator
def _genActMaxEq(m, g, t):
    if m.genType[g] == 1:
        return m.genActPower[g, t] <= m.genMax[g, t]
    elif m.genType[g] == 2:
        return m.genActPower[g, t] + m.genExcPower[g, t] == m.genMax[g, t]
    return default_behaviour

# Lower limit for the generator, type 1 controllable
def _genActMinEq(m, g, t):
    if m.genType[g] == 1:
        return m.genActPower[g, t] >= m.genMin[g] * m.genXo[g, t]
    return default_behaviour

# Limit for controllable load
def _loadRedActEq(m, l, t):
    return m.loadRedActPower[l, t] <= m.loadRedMax[l, t]

# Limits for on-off load    
def _loadCutActEq(m, l, t):
    return m.loadCutActPower[l, t] == m.loadCutMax[l, t] * m.loadXo[l, t]

# Limits for load ENS (a extra load served for the rest of available load)
def _loadENSEq(m, l, t):
    return m.loadENS[l, t] + m.loadRedActPower[l, t] + m.loadCutActPower[l, t] <= m.loadMax[l, t]

# Battery discharge limit 
def _storDchRateEq(m, s, t):
    return m.storDischarge[s, t] <= m.storDchMax[s, t] * m.storDchXo[s, t]

# Battery charge limit
def _storChRateEq(m, s, t):
    return m.storCharge[s, t] <= m.storChMax[s, t] * m.storChXo[s, t]

# Battery energy limit 
def _storMaxEq(m, s, t):
    return m.storState[s, t] <= m.storMax[s]

# Battery energy limit considering the relaxa variable  
def _storRelaxEq(m, s, t):
    return m.storState[s, t] >= m.storMax[s] * m.storMin[s]  - m.storRelax[s, t]

# Energy balance in the battery
def _storBalanceEq(m, s, t):
    if t == m.t.first():
        return m.storState[s, t] == m.storMax[s] * m.storStart[s] + m.storCharge[s, t] * m.storChEff[s] * 24/m.t.last() - m.storDischarge[s, t] * 24/m.t.last() / m.storDchEff[s]
    elif t > m.t.first():
        return m.storState[s, t] == m.storState[s, t - 1] + m.storCharge[s, t] * m.storChEff[s] * 24/m.t.last() - m.storDischarge[s, t] * 24/m.t.last() / m.storDchEff[s]
    return default_behaviour

# Binary limits for the battery
def _storChDchEq(m, s, t):
    return m.storChXo[s, t] + m.storDchXo[s, t] <= 1

# Battery charging power schedulled considering a relaxa variable
def _StorFollowScheduleChEq(m, s, t):
    return  (m.storCharge[s, t] - m.storScheduleChRelax[s, t]) == (m.storScheduleCh[s, t])

# Battery discharging power schedulled considering a relaxa variable
def _StorFollowScheduleDchEq(m, s, t):
    #Aqui ele pode tanto descarregar mais ou menos que o previsto nao?
    return  (m.storDischarge[s, t] - m.storScheduleDchRelax[s, t]) == (m.storScheduleDch[s, t])

# EV power discharge limit
def _v2gDchRateEq(m, v, t):
    return m.v2gDischarge[v, t] <= m.v2gDchMax[v, t] * m.v2gDchXo[v, t]

# EV power charge limit
def _v2gChRateEq(m, v, t):
    return m.v2gCharge[v, t] <= m.v2gChMax[v, t] * m.v2gChXo[v, t]

# Energy in the battery limit
def _v2gMaxEq(m, v, t):
    return m.v2gState[v, t] <= m.v2gMax[v]

def _v2gRelaxEq_Old(m, v, t):
    if m.v2gSchedule[v, t] == 1: #Quando chega ao ponto de carregamento tem que estar com maior SOC que o minimo
        return m.v2gState[v, t] >= m.v2gMin[v] - m.v2gRelax[v, t]
    elif t < m.t.last():  #Se nao estiver a carregar e nao for o ultimo timestep
        if (m.v2gSchedule[v, t] == 1) & (m.v2gSchedule[v, t + 1] == 0) & (m.v2gScheduleDepartureSOC[v, t] == 0):
            print("ESTRANHO")
            return m.v2gState[v, t] >= m.v2gMax[v] - m.v2gRelax[v, t]
    elif (m.v2gSchedule[v, t] == 1) & (m.v2gScheduleDepartureSOC[v, t] == 0) & (m.t == m.t.last()):
        print("ESTRANHO")
        return m.v2gState[v, t] >= m.v2gMax[v] - m.v2gRelax[v, t]
    return default_behaviour

# To validate the SOC requiered when the EV connects to the CS
def _v2gRelaxEq(m, v, t):
    if m.v2gSchedule[v, t] == 1: #Quando chega ao ponto de carregamento tem que estar com maior SOC que o minimo
        if m.v2gScheduleDepartureSOC[v, t] == 0:
            return m.v2gState[v, t] >= m.v2gMin[v] - m.v2gRelax[v, t]
        else:
            return m.v2gState[v, t] >= m.v2gScheduleDepartureSOC[v, t] - m.v2gRelax[v, t]
    else:
        return default_behaviour
    

# Energy balance in the EV battery    
def _v2gStateEq(m, v, t):
    if m.v2gSchedule[v, t] == 0: # If vehicle is not scheduled
        return m.v2gState[v, t] == 0
    elif (m.v2gSchedule[v, t] == 1) & (t == m.t.first()): # If vehicle is scheduled and it is the first time step
        return m.v2gState[v, t] == m.v2gScheduleArrivalSOC[v, t] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
    elif t > 1: # If not the first time step
        if (m.v2gSchedule[v, t - 1] == 1) & (m.v2gSchedule[v, t] == 1): # If was and is currently connected
            return m.v2gState[v, t] == m.v2gState[v, t - 1] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
        elif (m.v2gSchedule[v, t - 1] == 0) & (m.v2gSchedule[v, t] == 1): # If became connected
            return m.v2gState[v, t] == m.v2gScheduleArrivalSOC[v, t] + m.v2gCharge[v, t] * m.v2gChEff[v] * 24/m.t.last() - m.v2gDischarge[v, t] * 24/m.t.last() / m.v2gDchEff[v]
    return default_behaviour


# Binary limit for the EV battery
def _v2gChDchEq(m, v, t):
    #return m.v2gCharge[v, t] + m.v2gDischarge[v, t] <= 1
    return m.v2gChXo[v, t] + m.v2gDchXo[v, t] <= 1

# EV charge power limit considering relax variable
def _v2gFollowScheduleChEq(m, v, t):
    return  (m.v2gCharge[v, t] - m.v2gScheduleChRelax[v, t]) == (m.v2gScheduleCh[v, t])

# EV discharge power limit considering relax variable
def _v2gFollowScheduleDchEq(m, v, t):
    return  (m.v2gDischarge[v, t] - m.v2gScheduleDchRelax[v, t]) == (m.v2gScheduleDch[v, t])

# Limit for the CS power
def _csMaxEq(m, c, t):
    return m.csCharge[c, t] <= m.csMax[c]

# Limit for the CS discharge power 
def _csMinEq(m, c, t):
    return m.csCharge[c, t] >= -m.csMin[c]

# CS power considering charge and discharge power
def _csPowerEq(m, c, t):
    temp_val = []
    for v in np.arange(1, m.v2g.last() + 1):
        if m.csSchedule[c, v, t] > 0:# To validate if the CS is being used 
            temp_val.append(m.v2gCharge[v, t] - m.v2gDischarge[v, t])
    return m.csCharge[c, t] == sum(temp_val)

# CS power considering charge and discharge power considering the efficiences 
def _csNetChargeEq(m, c, t):
    temp_val = []
    for v in np.arange(1, m.v2g.last() + 1):
        if m.csSchedule[c, v, t] > 0:
            temp_val.append(m.v2gCharge[v, t] * m.v2gChEff[v] - m.v2gDischarge[v, t] / m.v2gDchEff[v])
    return m.csNetCharge[c, t] == sum(temp_val)

def _balanceEq(m, t):
    temp_gens = sum([m.genActPower[g, t] - m.genExcPower[g, t]
                     for g in np.arange(1, m.gen.last() + 1)])

    temp_loads = sum([m.loadMax[l, t] - m.loadRedActPower[l, t] - m.loadCutActPower[l, t] - m.loadENS[l, t]
                      for l in np.arange(1, m.loads.last() + 1)])

    temp_stor = sum([m.storCharge[s, t] - m.storDischarge[s, t]
                     for s in np.arange(1, m.stor.last() + 1)])

    temp_v2g = sum([m.v2gCharge[v, t] - m.v2gDischarge[v, t]
                    for v in np.arange(1, m.v2g.last() + 1)])

    #temp_cs = sum(m.csNetCharge[:, t])
    temp_cs = sum([m.csCharge[c, t] for c in np.arange(1, m.cs.last() + 1)])
    
    #return temp_gens - temp_loads - temp_stor - temp_v2g - temp_cs + m.imports[t] - m.exports[t] == 0
    return temp_gens - temp_loads - temp_stor - temp_cs + m.imports[t] - m.exports[t] == 0

def _objFn(m):

    temp_gens = sum([m.genActPower[g, t] * m.genActCost[g, t] + m.genExcPower[g, t] * m.genExcCost[g, t]
                     for t in np.arange(m.t.first(), m.t.last() + 1) for g in np.arange(1, m.gen.last() + 1)])

    temp_loads = sum([m.loadRedActPower[l, t] * m.loadRedCost[l, t] +\
                      m.loadCutActPower[l, t] * m.loadCutCost[l, t] +\
                      m.loadENS[l, t] * m.loadENSCost[l, t]
                      for t in np.arange(m.t.first(), m.t.last() + 1) for l in np.arange(1, m.loads.last() + 1)])
    #Não tinha o termos de relaxamento adicionado
    #Porque é q carregar vem com sinal negativo para a funcao de custo? [  ALTEREI  ]
    temp_stor = sum([m.storDischarge[s, t] * m.storDchCost[s, t] +\
                     m.storCharge[s, t] * m.storChCost[s, t] +\
                     (m.storScheduleChRelax[s, t])**2 * 2e-3 +\
                     (m.storScheduleDchRelax[s, t])**2 * 2e-3 +\
                     m.storRelax[s, t] *  1
                     for t in np.arange(m.t.first(), m.t.last() + 1) for s in np.arange(1, m.stor.last() + 1)])

    temp_v2g = sum([m.v2gDischarge[v, t] * m.v2gDchCost[v, t] + \
                    m.v2gCharge[v, t] * m.v2gChCost[v, t] +\
                    (m.v2gScheduleChRelax[v, t])**2 * 0.01 +\
                     (m.v2gScheduleDchRelax[v, t])**2 * 0.01 +\
                    m.v2gRelax[v, t] * 200
                    for t in np.arange(m.t.first(), m.t.last() + 1) for v in np.arange(1, m.v2g.last() + 1)])

    temp_rest = sum([m.imports[t] * m.impCost[t] - m.exports[t] * m.expCost[t]
                     for t in np.arange(m.t.first(), m.t.last() + 1)])

    return temp_gens + temp_loads + temp_stor + temp_v2g + temp_rest

def creat_model(data, i=1, end = 25):
    my_model = pe.ConcreteModel()
    my_model.t = pe.Set(initialize=np.arange(i, end),
                 doc='Time periods')
    return my_model

def def_import_export(data, model, i = 1):    
    model.impMax = pe.Param(model.t,
                            initialize=convert_to_dictionary(data.peers['import_contracted_p_max'][0, i-1:model.t.last()], t_start=i-1),
                            doc='Maximum import power')
    model.expMax = pe.Param(model.t,
                            initialize=convert_to_dictionary(data.peers['export_contracted_p_max'][0, i-1:model.t.last()], t_start=i-1),
                            doc='Maximum export power')
    model.impCost = pe.Param(model.t,
                            initialize=convert_to_dictionary(data.peers['buy_price'][0, i-1:model.t.last()], t_start=i-1),
                            doc='Import cost')
    model.expCost = pe.Param(model.t,
                            initialize=convert_to_dictionary(data.peers['sell_price'][0, i-1:model.t.last()], t_start=i-1),
                            doc='Export cost')

    # Variables
    model.imports = pe.Var(model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Imported power')
    model.exports = pe.Var(model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Exported power')
    
    
    model.impMaxEq = pe.Constraint(model.t, rule=_impMaxEq,
                                doc='Maximum import power')
    model.expMaxEq = pe.Constraint(model.t, rule=_expMaxEq,
                                doc='Maximum export power')
    
    return model

def def_gen(data, model, i = 1):
    model.gen = pe.Set(initialize=np.arange(1, data.generator['p_forecast'].shape[0] + 1),
                    doc='Number of generators')
    model.genType = pe.Param(model.gen,
                            initialize=convert_to_dictionary(data.generator['type_generator'].astype(int)),
                            doc='Types of generators')
    model.genMin = pe.Param(model.gen,
                            initialize=convert_to_dictionary(data.generator['p_min']),
                            doc='Minimum power generation')
    model.genMax = pe.Param(model.gen, model.t,
                            initialize=convert_to_dictionary(data.generator['p_forecast'][:, i-1:model.t.last()], t_start=i-1),
                            doc='Forecasted power generation')
    model.genActCost = pe.Param(model.gen, model.t,
                                initialize=convert_to_dictionary(data.generator['cost_parameter_b'][:, i-1:model.t.last()], t_start=i-1),
                                doc='Active power generation cost')
    model.genExcCost = pe.Param(model.gen, model.t,
                                initialize=convert_to_dictionary(data.generator['cost_nde'][:, i-1:model.t.last()], t_start=i-1),
                                doc='Excess power generation cost')

    # Variables
    model.genActPower = pe.Var(model.gen, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Active power generation')
    model.genExcPower = pe.Var(model.gen, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Excess power generation')
    model.genXo = pe.Var(model.gen, model.t, within=pe.Binary, initialize=0,
                        doc='Generation on/off')

    # Constraints
    model.genActMaxEq = pe.Constraint(model.gen, model.t, rule=_genActMaxEq,
                                    doc='Maximum active power generation')
    model.genActMinEq = pe.Constraint(model.gen, model.t, rule=_genActMinEq,
                                    doc='Minimum active power generation')
    return model

def def_loads(data, model, i = 1):
    model.loads = pe.Set(initialize=np.arange(1, data.load['p_forecast'].shape[0] + 1),
                        doc='Number of loads')
    model.loadMax = pe.Param(model.loads, model.t,
                            initialize=convert_to_dictionary(data.load['p_forecast'][:, i-1:model.t.last()] * 5, t_start=i-1),
                            doc='Forecasted power consumption')
    model.loadRedMax = pe.Param(model.loads, model.t,
                                initialize=convert_to_dictionary(data.load['p_forecast'][:, i-1:model.t.last()] * 0.5, t_start=i-1 ),
                                doc='Maximum power reduction')
    model.loadCutMax = pe.Param(model.loads, model.t,
                                initialize=convert_to_dictionary(data.load['p_forecast'][:, i-1:model.t.last()] * 0.5, t_start=i-1),
                                doc='Maximum power cut')
    model.loadRedCost = pe.Param(model.loads, model.t,
                                initialize=convert_to_dictionary(data.load['cost_reduce'][:, i-1:model.t.last()], t_start=i-1 ),
                                doc='Active power reduction cost')
    model.loadCutCost = pe.Param(model.loads, model.t,
                                initialize=convert_to_dictionary(data.load['cost_cut'][:, i-1:model.t.last()], t_start=i-1 ),
                                doc='Active power cut cost')
    model.loadENSCost = pe.Param(model.loads, model.t,
                                initialize=convert_to_dictionary(data.load['cost_ens'][:, i-1:model.t.last()], t_start=i-1 ),
                                doc='Energy not supplied cost')

    # Variables
    model.loadRedActPower = pe.Var(model.loads, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Active power reduction')
    model.loadCutActPower = pe.Var(model.loads, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Active power cut')
    model.loadENS = pe.Var(model.loads, model.t, within=pe.NonNegativeReals, initialize=0,
                        doc='Energy Not Supplied')
    model.loadXo = pe.Var(model.loads, model.t, within=pe.Binary, initialize=0,
                        doc='Load on/off')

    # Constraints
    
    model.loadReactEq = pe.Constraint(model.loads, model.t, rule=_loadRedActEq,
                                    doc='Maximum active power reduction')
    model.loadCutActEq = pe.Constraint(model.loads, model.t, rule=_loadCutActEq,
                                    doc='Maximum active power cut')
    model.loadENSEq = pe.Constraint(model.loads, model.t, rule=_loadENSEq,
                                    doc='Maximum energy not supplied')
    return model

def def_storage(data, model, forecast=True, schedule_storageCh=None, schedule_storageDch=None,
                i=1):
    model.stor = pe.Set(initialize=np.arange(1, data.storage['p_charge_limit'].shape[0] + 1),
                        doc='Number of storage units')
    model.storDchMax = pe.Param(model.stor, model.t,
                                initialize=convert_to_dictionary(data.storage['p_discharge_limit'][ :, i-1:model.t.last()], t_start=i-1),
                                doc='Maximum discharging power')
    model.storChMax = pe.Param(model.stor, model.t,
                            initialize=convert_to_dictionary(data.storage['p_charge_limit'][ :, i-1:model.t.last()], t_start=i-1),
                            doc='Maximum charging power')
    model.storMax = pe.Param(model.stor,
                            initialize=convert_to_dictionary(data.storage['energy_capacity']),
                            doc='Maximum energy capacity')
    model.storMin = pe.Param(model.stor,
                            initialize=convert_to_dictionary(data.storage['energy_min_percentage']),
                            doc='Minimum energy capacity')
    model.storStart = pe.Param(model.stor,
                            initialize=convert_to_dictionary(data.storage['initial_state']),
                            doc='Starting energy capacity')
    model.storDchEff = pe.Param(model.stor,
                                initialize=convert_to_dictionary(data.storage['discharge_efficiency']),
                                doc='Discharging efficiency')
    model.storChEff = pe.Param(model.stor,
                            initialize=convert_to_dictionary(data.storage['charge_efficiency']),
                            doc='Charging efficiency')
    model.storChCost = pe.Param(model.stor, model.t,
                                initialize=convert_to_dictionary(data.storage['charge_price'][ :, i-1:model.t.last()], t_start=i-1),
                                doc='Charging cost')
    model.storDchCost = pe.Param(model.stor, model.t,
                                initialize=convert_to_dictionary(data.storage['discharge_price'][ :, i-1:model.t.last()], t_start=i-1),
                                doc='Discharging cost')
    #Criate Schedule for storage
    if not forecast:
        model.storScheduleCh = pe.Param(model.stor, model.t,
                                        initialize=convert_to_dictionary(schedule_storageCh[ :, i-1:model.t.last()], t_start=i-1),
                                        doc='Storage schedule')
        model.storScheduleDch = pe.Param(model.stor, model.t,
                                        initialize=convert_to_dictionary(schedule_storageDch[ :, i-1:model.t.last()], t_start=i-1),
                                        doc='Storage schedule')
    # Variables
    model.storState = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='State of charge')
    model.storCharge = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Charging power')
    model.storDischarge = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Discharging power')
    model.storRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxation variable')
    model.storChXo = pe.Var(model.stor, model.t, within=pe.Binary, initialize=0,
                            doc='Charging on/off')
    model.storDchXo = pe.Var(model.stor, model.t, within=pe.Binary, initialize=0,
                            doc='Discharging on/off')
    model.storScheduleChRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxtion variable for following schedule of charging')
    model.storScheduleDchRelax = pe.Var(model.stor, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxtion variable for following schedule of discharging')
    # Constraints
    model.storDchRateEq = pe.Constraint(model.stor, model.t, rule=_storDchRateEq,
                                        doc='Maximum discharging rate')
    model.storChRateEq = pe.Constraint(model.stor, model.t, rule=_storChRateEq,
                                    doc='Maximum charging rate')
    model.storMaxEq = pe.Constraint(model.stor, model.t, rule=_storMaxEq,
                                    doc='Maximum energy capacity')
    model.storRelaxEq = pe.Constraint(model.stor, model.t, rule=_storRelaxEq,
                                    doc='Relaxation variable')
    model.storBalanceEq = pe.Constraint(model.stor, model.t, rule=_storBalanceEq,
                                        doc='Energy balance')
    model.storChDchEq = pe.Constraint(model.stor, model.t, rule=_storChDchEq,
                                    doc='Charging and discharging are mutually exclusive')
    if not forecast:
        model.storFollowScheduleChEq = pe.Constraint(model.stor, model.t, rule=_StorFollowScheduleChEq,
                                            doc='Charging and discharging are Schedule')
        model.storFollowScheduleDchEq = pe.Constraint(model.stor, model.t, rule=_StorFollowScheduleDchEq,
                                            doc='Charging and discharging are Schedule')
    return model

def def_v2g(data, model, forecast=True, schedule_v2gCh=None, schedule_v2gDch=None, i=1):
    model.v2g = pe.Set(initialize=np.arange(1, data.vehicle['p_charge_max'].shape[0] + 1),
                    doc='Number of EVs')
    model.v2gDchMax = pe.Param(model.v2g, model.t,
                            initialize=convert_to_dictionary((data.vehicle['p_discharge_max'] *
                                                                data.vehicle['schedule'].transpose()).transpose()[:, i-1:model.t.last()], t_start=i-1),
                            doc='Maximum scheduled discharging power')
    model.v2gChMax = pe.Param(model.v2g, model.t,
                            initialize=convert_to_dictionary((data.vehicle['p_charge_max'] *
                                                                data.vehicle['schedule'].transpose()).transpose()[:, i-1:model.t.last()], t_start=i-1),
                            doc='Maximum scheduled charging power')
    model.v2gDchEff = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(data.vehicle['discharge_efficiency']),
                            doc='Discharging efficiency')
    model.v2gChEff = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(data.vehicle['charge_efficiency']),
                            doc='Charging efficiency')
    model.v2gMax = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(data.vehicle['e_capacity_max']),
                            doc='Maximum energy capacity')
    model.v2gMin = pe.Param(model.v2g,
                            initialize=convert_to_dictionary(data.vehicle['e_capacity_max'] * data.vehicle['min_technical_soc']),
                            doc='Minimum energy capacity')
    model.v2gSchedule = pe.Param(model.v2g, model.t,
                                initialize=convert_to_dictionary(data.vehicle['schedule'][ :, i-1:model.t.last()], t_start=i-1),
                                doc='Vehicle schedule')
    model.v2gScheduleArrivalSOC = pe.Param(model.v2g, model.t,
                                        initialize=convert_to_dictionary(data.vehicle['schedule_arrival_soc'][ :, i-1:model.t.last()], t_start=i-1),
                                        doc='Vehicle schedule arrival SOC')
    model.v2gScheduleDepartureSOC = pe.Param(model.v2g, model.t,
                                            initialize=convert_to_dictionary(data.vehicle['schedule_departure_soc'][ :, i-1:model.t.last()], t_start=i-1),
                                            doc='Vehicle schedule required')
    model.v2gChCost = pe.Param(model.v2g, model.t,
                            initialize=convert_to_dictionary((np.ones(data.vehicle['schedule'].shape).transpose() *\
                                                                data.vehicle['charge_price'][:, 0]).transpose()[ :, i-1:model.t.last()], t_start=i-1),
                            doc='Charging cost')
    model.v2gDchCost = pe.Param(model.v2g, model.t,
                                initialize=convert_to_dictionary((np.ones(data.vehicle['schedule'].shape).transpose() *\
                                                                data.vehicle['discharge_price'][:, 0]).transpose()[ :, i-1:model.t.last()], t_start=i-1),
                                doc='Discharging cost')
    if not forecast:
        model.v2gScheduleCh = pe.Param(model.v2g, model.t,
                                        initialize=convert_to_dictionary(schedule_v2gCh[ :, i-1:model.t.last()], t_start=i-1),
                                        doc='Storage schedule')
        model.v2gScheduleDch = pe.Param(model.v2g, model.t,
                                        initialize=convert_to_dictionary(schedule_v2gDch[ :, i-1:model.t.last()], t_start=i-1),
                                        doc='Storage schedule')
    
    # Variables
    model.v2gCharge = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Charging power')
    model.v2gDischarge = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                                doc='Discharging power')
    model.v2gState = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='State of charge')
    model.v2gRelax = pe.Var(model.v2g, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Relaxation variable')
    model.v2gChXo = pe.Var(model.v2g, model.t, within=pe.Binary, initialize=0,
                        doc='Charging on/off')
    model.v2gDchXo = pe.Var(model.v2g, model.t, within=pe.Binary, initialize=0,
                            doc='Discharging on/off')
    model.v2gScheduleChRelax = pe.Var(model.v2g, model.t, within=pe.Reals, initialize=0,
                            doc='Relaxtion variable for following schedule of charging')
    model.v2gScheduleDchRelax = pe.Var(model.v2g, model.t, within=pe.Reals, initialize=0,
                            doc='Relaxtion variable for following schedule of discharging')

    # Constraints
    model.v2gDchRateEq = pe.Constraint(model.v2g, model.t, rule=_v2gDchRateEq,
                                    doc='Maximum discharging rate')
    model.v2gChRateEq = pe.Constraint(model.v2g, model.t, rule=_v2gChRateEq,
                                    doc='Maximum charging rate')
    model.v2gMaxEq = pe.Constraint(model.v2g, model.t, rule=_v2gMaxEq,
                                doc='Maximum energy capacity')
    model.v2gRelaxEq = pe.Constraint(model.v2g, model.t, rule=_v2gRelaxEq,
                                    doc='Relaxation variable')
    model.v2gStateEq = pe.Constraint(model.v2g, model.t, rule=_v2gStateEq,
                                    doc='State of charge')
    model.v2gChDchEq = pe.Constraint(model.v2g, model.t, rule=_v2gChDchEq,
                                    doc='Charging and discharging cannot occur simultaneously')
    if not forecast:
        model.v2gFollowScheduleChEq = pe.Constraint(model.v2g, model.t, rule=_v2gFollowScheduleChEq,
                                            doc='Charging are Schedule')
        model.v2gFollowScheduleDchEq = pe.Constraint(model.v2g, model.t, rule=_v2gFollowScheduleDchEq,
                                            doc='Discharging are Schedule')
    
    return model

def def_charging_stations(data, model, i=1, init=False):
    model.cs = pe.Set(initialize=np.arange(1, data.charging_station['p_charge_limit'].shape[0] + 1),
                    doc='Number of charging stations')
    model.csMax = pe.Param(model.cs, initialize=convert_to_dictionary(data.charging_station['p_charge_max']),
                        doc='Maximum charging power')
    model.csMin = pe.Param(model.cs, initialize=convert_to_dictionary(data.charging_station['p_discharge_max']),
                        doc='Maximum discharging power')
    model.csChEff = pe.Param(model.cs, initialize=convert_to_dictionary(
        data.charging_station['charge_efficiency'] * 0.01),
                            doc='Charging efficiency')
    model.csDchEff = pe.Param(model.cs, initialize=convert_to_dictionary(
        data.charging_station['discharge_efficiency'] * 0.01),
                            doc='Discharging efficiency')
    model.csSchedule = pe.Param(model.cs, model.v2g, model.t, initialize=convert_to_dictionary(
        data.vehicle['schedule_cs_usage'][:, :, i-1:model.t.last()], t_start=i-1),
                                doc='Charging station schedule')

    # Variables
    model.csCharge = pe.Var(model.cs, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Charging power')
    model.csNetCharge = pe.Var(model.cs, model.t, within=pe.NonNegativeReals, initialize=0,
                            doc='Net charging power')

    # Constraints
    model.csMaxEq = pe.Constraint(model.cs, model.t, rule=_csMaxEq,
                                doc='Maximum charging power')
    model.csMinEq = pe.Constraint(model.cs, model.t, rule=_csMinEq,
                                doc='Maximum discharging power')
    model.csPowerEq = pe.Constraint(model.cs, model.t, rule=_csPowerEq,
                                    doc='Charging station power')
    model.csNetChargeEq = pe.Constraint(model.cs, model.t, rule=_csNetChargeEq,
                                        doc='Net charging power')

    return model



def plot_profile(
    result_genActPower,
    result_storDchActPower,
    result_v2gDchActPower,
    result_loadRedActPower,
    result_loadCutActPower,
    result_loadENS,
    result_pimp,
    model,
    Data,
    result_genExcActPower,
    result_storChActPower,
    result_v2gChActPower,
    save = False,
    path = None,
    name = None,
    graph_max = None,
    graph_step = None
):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Production
    y1_prod = sum([result_genActPower.values[i].astype(float) for i in range(model.gen.last())]) 
    y2_prod = sum([result_storDchActPower.values[i] for i in range(model.stor.last())]) 
    y3_prod = sum([result_v2gDchActPower.values[i] for i in range(model.v2g.last())]) 
    y4_prod = sum([result_loadRedActPower.values[i] for i in range(model.loads.last())]) 
    y5_prod = sum([result_loadCutActPower.values[i] for i in range(model.loads.last())]) 
    y6_prod = sum([result_loadENS.values[i] for i in range(model.loads.last())]) 
    
    y7_prod = result_pimp.values.reshape(model.t.last()-model.t.first()+1) 


    # Consumption
    y1_cons = np.sum(Data.get_data().load['p_forecast'][:, model.t.first()-1:model.t.last()]*5, axis=0, dtype=np.float64) 
    y2_cons = sum([result_genExcActPower.values[i] for i in range(model.gen.last())])
    y3_cons = sum([result_storChActPower.values[i] for i in range(model.stor.last())]) 
    y4_cons = sum([result_v2gChActPower.values[i] for i in range(model.v2g.last())]) 

    # Plot Production
    axs[0].fill_between(list(range(1, len(y1_prod)+1)), np.zeros(len(y1_prod)), y1_prod, color=project_colors_list[0], label="Generator Power Production")
    axs[0].fill_between(list(range(1, len(y2_prod)+1)), y1_prod, y1_prod + y2_prod, color=project_colors_list[1], label="BESS Discharging Power")
    axs[0].fill_between(list(range(1, len(y3_prod)+1)), y1_prod + y2_prod, y1_prod + y2_prod + y3_prod, color=project_colors_list[2], label="EV Discharging Power")
    axs[0].fill_between(list(range(1, len(y4_prod)+1)), y1_prod + y2_prod + y3_prod, y1_prod + y2_prod + y3_prod + y4_prod, color=project_colors_list[3], label="Load Reduction")
    axs[0].fill_between(list(range(1, len(y5_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, color=project_colors_list[4], label="Load Cut")
    axs[0].fill_between(list(range(1, len(y6_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, color=project_colors_list[5], label="Load ENS")
    axs[0].fill_between(list(range(1, len(y7_prod)+1)), y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod, y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod, color=project_colors_list[6], label="Grid Import Power")
    axs[0].set_ylim(0, max(1.1*np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod), 1.1*np.max(y1_cons + y2_cons + y3_cons + y4_cons)))
    
    if graph_max and graph_step:
        axs[0].set_ylim(0, graph_max)
        axs[0].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))
    
    axs[0].set_xlabel('Hour')
    axs[0].set_ylabel('Power [kW]')
    #axs[0].set_title('Production')
    axs[0].legend()

    # Plot Consumption
    axs[1].fill_between(list(range(1, len(y1_cons)+1)), np.zeros(len(y1_cons)), y1_cons, color=project_colors_list[0], label="Load Power Consumption")
    axs[1].fill_between(list(range(1, len(y2_cons)+1)), y1_cons, y1_cons + y2_cons, color=project_colors_list[1], label="Grid Export Power")
    axs[1].fill_between(list(range(1, len(y3_cons)+1)), y1_cons + y2_cons, y1_cons + y2_cons + y3_cons, color=project_colors_list[2], label="BESS Charging Power")
    axs[1].fill_between(list(range(1, len(y4_cons)+1)), y1_cons + y2_cons + y3_cons, y1_cons + y2_cons + y3_cons + y4_cons, color=project_colors_list[3], label="EV Charging Power")
    
    
    axs[1].set_ylim(0, max(1.1*np.max(y1_prod + y2_prod + y3_prod + y4_prod + y5_prod + y6_prod + y7_prod), 1.1*np.max(y1_cons + y2_cons + y3_cons + y4_cons)))
    if graph_max and graph_step:
        axs[1].set_ylim(0, graph_max)
        axs[1].yaxis.set_major_locator(ticker.MultipleLocator(graph_step))

        
    axs[1].set_xlabel('Hour')
    axs[1].set_ylabel('Power [kW]')
    #axs[1].set_title('Consumption')
    axs[1].legend()

    axs[0].set_xlim(1, model.t.last())
    axs[1].set_xlim(1, model.t.last())
    ticks = range(1, model.t.last() + 1)
    axs[0].set_xticks(ticks)
    axs[1].set_xticks(ticks)

    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    plt.tight_layout()
    
    if save:
        full_path = os.path.join(path, name)

        if not os.path.exists(path):
            os.makedirs(path)

        # Save the plot to the specified path
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def export_profile_to_excel(
    result_genActPower,
    result_storDchActPower,
    result_v2gDchActPower,
    result_loadRedActPower,
    result_loadCutActPower,
    result_loadENS,
    result_pimp,
    model,
    Data,
    result_genExcActPower,
    result_storChActPower,
    result_v2gChActPower,
    path,
    name
):
    # True Production
    y1_prod = sum([result_genActPower.values[i].astype(float) for i in range(model.gen.last())]) 
    y2_prod = sum([result_storDchActPower.values[i] for i in range(model.stor.last())])
    y3_prod = sum([result_v2gDchActPower.values[i] for i in range(model.v2g.last())]) 
    y4_prod = sum([result_loadRedActPower.values[i] for i in range(model.loads.last())]) 
    y5_prod = sum([result_loadCutActPower.values[i] for i in range(model.loads.last())]) 
    y6_prod = sum([result_loadENS.values[i] for i in range(model.loads.last())])
    y7_prod = result_pimp.values.reshape(model.t.last()-model.t.first()+1) 
    
    # True Consumption
    y1_cons = np.sum(Data.get_data().load['p_forecast'][:, model.t.first()-1:model.t.last()]*5, axis=0, dtype=np.float64) 
    y2_cons = sum([result_genExcActPower.values[i] for i in range(model.gen.last())])
    y3_cons = sum([result_storChActPower.values[i] for i in range(model.stor.last())]) 
    y4_cons = sum([result_v2gChActPower.values[i] for i in range(model.v2g.last())]) 
    
    # Creating Production Dataframe
    df_production = pd.DataFrame({
        'Time [h]': list(range(1, len(y1_prod) + 1)),
        'Generators': y1_prod,
        'Storage': y2_prod,
        'V2G Discharge': y3_prod,
        'Load Reduction': y4_prod,
        'Load Cut': y5_prod,
        'Load ENS': y6_prod,
        'Imports': y7_prod
    })
    
    # Creating Consumption Dataframe
    df_consumption = pd.DataFrame({
        'Time [h]': list(range(1, len(y1_cons) + 1)),
        'Load': y1_cons,
        'Gen Excess': y2_cons,
        'Storage': y3_cons,
        'V2G Charge': y4_cons
    })

    full_path = os.path.join(path, name)

    if not os.path.exists(path):
        os.makedirs(path)
    
    with pd.ExcelWriter(full_path, engine='openpyxl') as writer:
        df_production.to_excel(writer, sheet_name='Production', index=False)
        df_consumption.to_excel(writer, sheet_name='Consumption', index=False)

def store_results(hour, results_dict,
                  result_pimp, result_pexp, result_genActPower, result_genExcActPower, 
                  result_genXo, result_loadRedActPower, result_loadCutActPower, 
                  result_loadENS, result_loadXo, result_storEnerState, 
                  result_storDchActPower, result_storChActPower, result_storDchXo, 
                  result_storChXo, result_storScheduleChRelax, result_storScheduleDchRelax, 
                  result_v2gChActPower, result_v2gDchActPower, result_v2gEnerState, 
                  result_v2gRelax, result_v2gDchXo, result_v2gChXo, result_v2gScheduleChRelax, 
                  result_v2gScheduleDchRelax, result_csActPower, result_csActPowerNet):
    

    results_dict[hour] = {
        'importsInit': np.array(result_pimp)[0],
        'exportsInit': np.array(result_pexp)[0],
        'genActPowerInit': np.array(result_genActPower)[:, 0],
        'genExcPowerInit': np.array(result_genExcActPower)[:, 0],
        'genXoInit': np.array(result_genXo)[:, 0],
        'loadRedActPowerInit': np.array(result_loadRedActPower)[:, 0],
        'loadCutActPowerInit': np.array(result_loadCutActPower)[:, 0],
        'loadENSInit': np.array(result_loadENS)[:, 0],
        'loadXoInit': np.array(result_loadXo)[:, 0],
        'storStateInit': np.array(result_storEnerState)[:, 0],
        'storDischargeInit': np.array(result_storDchActPower)[:, 0],
        'storChargeInit': np.array(result_storChActPower)[:, 0],
        'storRelaxInit': np.array(result_storDchXo)[:, 0],
        'storChXoInit': np.array(result_storChXo)[:, 0],
        'storDchXoInit': np.array(result_storDchXo)[:, 0],
        'storScheduleChRelaxInit': np.array(result_storScheduleChRelax)[:, 0],
        'storScheduleDchRelaxInit': np.array(result_storScheduleDchRelax)[:, 0],
        'v2gChargeInit': np.array(result_v2gChActPower)[:, 0],
        'v2gDischargeInit': np.array(result_v2gDchActPower)[:, 0],
        'v2gStateInit': np.array(result_v2gEnerState)[:, 0],
        'v2gRelaxInit': np.array(result_v2gRelax)[:, 0],
        'v2gDchXoInit': np.array(result_v2gDchXo)[:, 0],
        'v2gChXoInit': np.array(result_v2gChXo)[:, 0],
        'v2gScheduleChRelaxInit': np.array(result_v2gScheduleChRelax)[:, 0],
        'v2gScheduleDchRelaxInit': np.array(result_v2gScheduleDchRelax)[:, 0],
        'csChargeInit': np.array(result_csActPower)[:, 0],
        'csNetChargeInit': np.array(result_csActPowerNet)[:, 0]
    }

    return results_dict

def metric01(pimp_ideal, pimp_real, ered_ideal, ered_real, ecut_ideal,
             ecut_real, ens_ideal, ens_real, a=1, b=1, c=1, d=1):
    metric = 0

    if np.sum(pimp_ideal) != 0:
        metric += (np.sum(pimp_ideal) - np.sum(pimp_real))**2/np.sum(pimp_ideal) * a
    if np.sum(ered_ideal) != 0:
        metric += (np.sum(ered_ideal) - np.sum(ered_real))**2/np.sum(ered_ideal) * b
    if np.sum(ecut_ideal) != 0: 
        metric += (np.sum(ecut_ideal) - np.sum(ecut_real))**2/np.sum(ecut_ideal) * c
    if np.sum(ens_ideal) != 0:
        metric += (np.sum(ens_ideal) - np.sum(ens_real))**2/np.sum(ens_ideal) * d

    return metric

def metric02_helper(a, b):
    n = 0
    d = 0
    if len(a.squeeze().shape) == 1:
        for x in range(a.shape[0]):
            n += np.abs(a[x] - b[x])
            d += a[x] + b[x]
            
    elif len(a.squeeze().shape) == 2:
        for x in range(a.shape[1]):
            for y in range(a.shape[0]):
                n += a[y][x] - b[y][x]
                d += a[y][x] + b[y][x]
    if d != 0:
        return n/d
    else:
        return 0
    
def metric02(pimp_ideal, pimp_real, ered_ideal, ered_real, ecut_ideal,
             ecut_real, ens_ideal, ens_real, a=1, b=1, c=1, d=1):
    #trocar ordem dos agumentos no metric02_helper
    metric = 0
    metric += a*metric02_helper(pimp_ideal, pimp_real)
    metric += b*metric02_helper(ered_ideal, ered_real)
    metric += c*metric02_helper(ecut_ideal, ecut_real)
    metric += d*metric02_helper(ens_ideal, ens_real)
  
    return metric

