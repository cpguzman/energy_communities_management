cplex_path = r"C:\Program Files\IBM\ILOG\CPLEX_Studio129\cplex\bin\x64_win64\cplex.exe"
import pyomo.environ as pyo
import pandas as pd
model = pyo.ConcreteModel("Energy Community")
#******************************************************************************
#                                    Data
#*****************************************************************************
General_Information_Data_excel = pd.ExcelFile(".\Dados\General_Information Data.xlsx")
df_P_Max_Imp = General_Information_Data_excel.parse("1 P Max_Imp (kW)",index_col=0)
dict_P_Max_Imp = df_P_Max_Imp.to_dict()
df_P_Max_Exp = General_Information_Data_excel.parse("2 P Max_Exp (kW)",index_col=0)
dict_P_Max_Exp = df_P_Max_Exp.to_dict()
Load_Data_excel =  pd.ExcelFile(".\Dados\Load_Data.xlsx")
df_Pload_Forecast = Load_Data_excel.parse("1 P Forecast (kW)",index_col=0)
Pload_Forecast_dict = df_Pload_Forecast.to_dict()
Generators_Data_excel =  pd.ExcelFile(".\Dados\Generators_Data.xlsx")
df_GenCharacteritics = Generators_Data_excel.parse("9 Characteritics D.",index_col=0)
GenCharacteritics_dict = df_GenCharacteritics.to_dict()
df_Pgen_Forecast = Generators_Data_excel.parse("1 P Forecast (kW)",index_col=0)
Pgen_Forecast_dict = df_Pgen_Forecast.to_dict()
df_GenCost = Generators_Data_excel.parse("3 Cost Parameter B (m.u.)",index_col=0)
Pgen_Cost_dict = df_GenCost.to_dict()
df_P_Imp_Price = General_Information_Data_excel.parse("3 Energy_Buy_Price (m.u.)",index_col=0)
dict_C_PImp_Price = df_P_Imp_Price.to_dict()
df_P_Exp_Price = General_Information_Data_excel.parse("4 Energy_Sell_Price (m.u.)",index_col=0)
dict_C_PExp_Price = df_P_Exp_Price.to_dict()
Storage_Data_excel =  pd.ExcelFile(".\Dados\Storage_Data.xlsx")
df_PstorageChargePrice = Storage_Data_excel.parse("3 Charge price (m.u)",index_col=0)
PstorageChargePrice_dict = df_PstorageChargePrice.to_dict()
df_PstorageDischargePrice = Storage_Data_excel.parse("4 Discharge price (m.u.)",index_col=0)
PstorageDischargePrice_dict = df_PstorageDischargePrice.to_dict()
df_PstorageChargeLimit = Storage_Data_excel.parse("1 P Charge Limit (kW)",index_col=0)
PstorageChargeLimit_dict = df_PstorageChargeLimit.to_dict()
df_PstorageDischargeLimit = Storage_Data_excel.parse("2 P Discharge Limit (kW)",index_col=0)
PstorageDischargeLimit_dict = df_PstorageDischargeLimit.to_dict()
df_StorageCharacteritics = Storage_Data_excel.parse("5 Characteritics D.",index_col=0)
StorageCharacteritics_dict = df_StorageCharacteritics.to_dict()
V2G_Data_excel =  pd.ExcelFile(".\Dados\V2G_Data.xlsx")
df_V2G_ExitRequired = V2G_Data_excel.parse("3 Energy Exit Required ",index_col=0)
V2G_ExitRequired_dict = df_V2G_ExitRequired.to_dict()
df_V2G_EinicialArriving = V2G_Data_excel.parse("2 Energy Inicial Arriving",index_col=0)
V2G_EinicialArriving_dict = df_V2G_EinicialArriving.to_dict()
df_V2G_Characteritics = V2G_Data_excel.parse("8 Characteritics D.",index_col=0)
V2G_Characteritics_dict = df_V2G_Characteritics.to_dict()
df_V2G_DischargePrice = V2G_Data_excel.parse("7 Disharge Price",index_col=0)
V2G_DischargePrice_dict = df_V2G_DischargePrice.to_dict()
df_V2G_ChargePrice = V2G_Data_excel.parse("6 Charge Price",index_col=0)
V2G_ChargePrice_dict = df_V2G_ChargePrice.to_dict()
df_V2G_ConexionStatus = V2G_Data_excel.parse("1 ConexionStatusV2G",index_col=0)
V2G_ConexionStatus_dict = df_V2G_ConexionStatus.to_dict()
#***********************************************************************************
#                                 Defining Sets
#***********************************************************************************
nW = 3
pi_w = [0.3, 0.3, 0.3]
model.SetSceneries = pyo.Set(initialize=list(range(nW)), doc='Set of Scenarios')
model.SetTimeIntervals = pyo.Set(initialize=df_Pload_Forecast.columns, doc='Set of time intervals')
model.LoadSet = pyo.Set(initialize=df_Pload_Forecast.index, doc='Set of Loads')
model.GenSet = pyo.Set(initialize=df_Pgen_Forecast.index, doc='Set of Generatos')
model.BatterySet = pyo.Set(initialize=df_PstorageChargeLimit.index, doc='Set of Energy Storage System')
model.V2G_Set = pyo.Set(initialize=df_V2G_Characteritics.index, doc='Set of Electric Vehicles')
#***********************************************************************************
#                              Defining the parameter
#***********************************************************************************
@model.Param(model.GenSet)
def Param_P_MaxGen_g(model, gen):
  return GenCharacteritics_dict['6 P Max. (kW)'][gen]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamC_UpGenCost_g_t(model, generator, time):
  return 1.2 * Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def ParamC_DownGenCost_g_t(model, generator, time):
  return 0.8 * Pgen_Cost_dict[time][generator]

@model.Param(model.SetTimeIntervals)
def Param_P_MaxExp_t(model, time):
  return dict_P_Max_Exp[time]['PCC_0']

@model.Param(model.LoadSet, model.SetTimeIntervals, within=pyo.NonNegativeReals)
def Param_Pld_l_t(model, load, time):
  return Pload_Forecast_dict[time][load]

@model.Param(model.SetTimeIntervals)
def Param_P_MaxImp_t(model, time):
  return dict_P_Max_Imp[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_ImpPrice_t(model, time):
  return dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_c_UpImp_t(model, time):
  return 1.1 * dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_c_DownImp_t(model, time):
  return 0.9 * dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_UpImpPrice_t(model, time):
  return 1.2 * dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_DownImpPrice_t(model, time):
  return 0.8 * dict_C_PImp_Price[time]['PCC_0']

@model.Param(model.SetTimeIntervals)
def Param_C_ExpPrice_t(model, time):
  return dict_C_PExp_Price[time]['PCC_0']

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gent_g_t(model, generator, time):
  return Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gen_Up_g_t(model, generator, time):
  return 1.1 * Pgen_Cost_dict[time][generator]

@model.Param(model.GenSet, model.SetTimeIntervals)
def Param_c_Gen_Down_g_t(model, generator, time):
  return 0.9 * Pgen_Cost_dict[time][generator]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def Param_c_St_Ch_s_t(model, s, time):
  return  PstorageChargePrice_dict[time][s]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def Param_c_St_Dch_s_t(model, s, time):
  return  PstorageDischargePrice_dict[time][s]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def Param_c_St_minRlx_s_t(model, s, time):
  return  PstorageChargePrice_dict[time][s]

# Pv2gDischargeMax_v parameters: Contains the maximum power of discharging of each electrical vehicle
@model.Param(model.V2G_Set)
def Pv2gDischargeMax_v(model, v2g):
  return V2G_Characteritics_dict["7 P Discharge Max (kW)"][v2g]

# Pv2gChargeMax_v parameters: Contains the maximum power of charging of each electrical vehicle
@model.Param(model.V2G_Set)
def Pv2gChargeMax_v(model, v2g):
  return V2G_Characteritics_dict["6 P Charge Max (kW)"][v2g]

# model.EstorageMax_b: Contains the maximum energy storage capacity of each battery
@model.Param(model.BatterySet)
def EstorageMax_b(model, storage):
  return StorageCharacteritics_dict["6 Energy Capacity (kVAh)"][storage]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def ParamPstorageChargeLimit_b_t(model, storage, time):
  return PstorageChargeLimit_dict[time][storage]

@model.Param(model.BatterySet, model.SetTimeIntervals)
def ParamPstorageDischargeLimit_b_t(model, storage, time):
  return PstorageDischargeLimit_dict[time][storage]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def Param_c_EV_Dch_e_t(model, v2g, time):
  return  V2G_DischargePrice_dict[time][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def Param_c_EV_Ch_e_t(model, v2g, time):
  return  V2G_ChargePrice_dict[time][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals)
def Param_c_EV_minRlx_e_t(model, v2g, time):
  return  V2G_ChargePrice_dict[time][v2g]

# Ev2gMax_v parameters: Contains the maximum capacity of each battery of each electrical Vehicle
@model.Param(model.V2G_Set)
def Ev2gMax_v(model, v2g):
  return V2G_Characteritics_dict["5 E Capacity Max (kWh)"][v2g]

# Ev2gInitial_v_t parameters: Contains the initial Energy of each battery of each electrical Vehicle at arriving
@model.Param(model.V2G_Set, model.SetTimeIntervals)
def Ev2gInitial_v_t(model, v2g, time):
  return V2G_EinicialArriving_dict[time][v2g]

@model.Param(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries)
def XconexionStatus_v_t(model, v2g, time, w):
  return V2G_ConexionStatus_dict[time][v2g]

# ParamEv2gExitRequired_v_t: Contains the exit Energy of each battery required of each electrical Vehicle at exit
@model.Param(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries)
def ParamEv2gExitRequired_v_t(model, v2g, time, w):
  return V2G_ExitRequired_dict[time][v2g]
#****************************************************************************************
#                            Defining Decision Variables
#****************************************************************************************
model.VarP_OpGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

def UpGen_bounds(model, gen, time):
    lower_bound = 0
    upper_bound = model.Param_P_MaxGen_g[gen]
    return (lower_bound, upper_bound)

model.VarR_UpGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, bounds = UpGen_bounds, domain = pyo.NonNegativeReals, initialize=0)

def DownGen_bounds(model, gen, time):
    lower_bound = 0
    upper_bound = model.Param_P_MaxGen_g[gen]
    return (lower_bound, upper_bound)

model.VarR_DownGen_g_t = pyo.Var(model.GenSet, model.SetTimeIntervals, bounds = DownGen_bounds, domain = pyo.NonNegativeReals, initialize=0)

def R_Up_Import_bounds(model, time):
  lower_bound = 0
  upper_bound = model.Param_P_MaxImp_t[time]
  return (lower_bound, upper_bound)

model.VarR_UpImp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, bounds = R_Up_Import_bounds, initialize=0)

def R_Down_Import_bounds(model, time):
  lower_bound = 0
  upper_bound = model.Param_P_MaxImp_t[time]
  return (lower_bound, upper_bound)

model.VarR_DownImp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, bounds = R_Down_Import_bounds, initialize=0)

model.Var_r_UpGen_g_t_w = pyo.Var(model.GenSet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_DownGen_g_t_w = pyo.Var(model.GenSet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.VarP_Op_imp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_Up_imp_t_w = pyo.Var(model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.Var_r_Down_imp_t_w = pyo.Var(model.SetTimeIntervals, model.SetSceneries, domain = pyo.NonNegativeReals, initialize=0)

model.VarP_Op_Exp_t = pyo.Var(model.SetTimeIntervals, domain = pyo.NonNegativeReals, initialize=0)

model.X_imp_t = pyo.Var(model.SetTimeIntervals, domain=pyo.Binary, bounds=(0, 1), initialize=0)
model.X_exp_t = pyo.Var(model.SetTimeIntervals, domain=pyo.Binary, bounds=(0, 1), initialize=0)
model.XstorageCharge_b_t = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.Binary, initialize = 0)
model.XstorageDischarge_b_t = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, domain = pyo.Binary, initialize = 0)

model.Xv2gCharge_v_t = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, domain = pyo.Binary, initialize = 0)
model.Xv2gDischarge_v_t = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, domain = pyo.Binary, initialize = 0)


def EnergyV2G_bounds(model, v2g, time, w):
  lower_bound = 0
  upper_bound = model.Ev2gMax_v[v2g]
  return (lower_bound, upper_bound)

model.VarEnergyV2G_v_t = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries , bounds=EnergyV2G_bounds)

def EnergyStorage_bounds(model, storage, time, w):
  lower_bound = model.EstorageMax_b[storage] * StorageCharacteritics_dict["7 Energy Min (%)"][storage] / 100
  upper_bound = model.EstorageMax_b[storage]
  return (lower_bound, upper_bound)
model.VarEnergyStorage_b_t = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, bounds=EnergyStorage_bounds)


def storageCharge_bounds(model, storage, time, w):
  lower_bound = 0
  upper_bound = model.ParamPstorageChargeLimit_b_t[storage, time]
  return (lower_bound, upper_bound)
model.VarPstorageCharge_b_t = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, bounds=storageCharge_bounds)

model.VarP_St_minRlx_s_t_w = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, bounds=storageCharge_bounds)

def storageDischarge_bounds(model, storage, time, w):
  lower_bound = 0
  upper_bound = model.ParamPstorageDischargeLimit_b_t[storage, time]
  return (lower_bound, upper_bound)
model.VarPstorageDischarge_b_t = pyo.Var(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, bounds=storageDischarge_bounds)

def v2gDischarge_bounds(model, v2g, time, w):
  lower_bound = 0
  upper_bound = model.Pv2gDischargeMax_v[v2g]
  return (lower_bound, upper_bound)
model.VarPv2gDischarge_v_t = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, bounds=v2gDischarge_bounds)

def v2gCharge_bounds(model, v2g, time, w):
  lower_bound = 0
  upper_bound = model.Pv2gChargeMax_v[v2g]
  return (lower_bound, upper_bound)
model.VarPv2gCharge_v_t = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, bounds=v2gCharge_bounds, domain = pyo.NonNegativeReals, initialize = 0)

#model.VarP_Ev_minRlx_e_t_w = pyo.Var(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, bounds=v2gCharge_bounds, domain = pyo.NonNegativeReals, initialize = 0)

def rule_power_reserve_Up(model, gen, time, w):
  return model.Var_r_UpGen_g_t_w[gen, time, w] <= model.VarR_UpGen_g_t[gen, time]
model.power_reserve_up_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Up)

def rule_power_reserve_Down(model, gen, time, w):
  return model.Var_r_DownGen_g_t_w[gen, time, w] <= model.VarR_DownGen_g_t[gen, time]
model.power_reserve_Down_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down)

def rule_power_reserve_Up_import(model,time, w):
  return model.Var_r_Up_imp_t_w[time, w] <= model.VarR_UpImp_t[time]
model.power_reserve_Up_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Up_import)

def rule_power_reserve_Down_import(model,time, w):
  return model.Var_r_Down_imp_t_w[time, w] <= model.VarR_DownImp_t[time]
model.power_reserve_Down_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down_import)

def rule_gen_power_max(model, gen, time, w):
  return model.VarP_OpGen_g_t[gen, time] + model.Var_r_UpGen_g_t_w[gen, time, w] <= model.Param_P_MaxGen_g[gen]
model.gen_power_max_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_gen_power_max)

def rule_gen_power_min(model, gen, time, w):
  return model.VarP_OpGen_g_t[gen, time] - model.Var_r_Down_imp_t_w[time, w] >= 0
model.gen_power_min_constraint = pyo.Constraint(model.GenSet, model.SetTimeIntervals, model.SetSceneries, rule=rule_gen_power_min)

def rule_power_imp(model, time, w):
  return model.VarP_Op_imp_t[time] + model.Var_r_Up_imp_t_w[time, w] <= model.Param_P_MaxImp_t[time]
model.power_imp_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_imp)

def rule_power_reserve_Down_import_Op(model,time, w):
  return model.VarP_Op_imp_t[time] - model.Var_r_Down_imp_t_w[time, w] >= 0
model.power_reserve_Down_import_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_power_reserve_Down_import_Op)

def rule_power_export(model, time):
  return model.VarP_Op_Exp_t[time] <= model.Param_P_MaxExp_t[time]
model.power_export_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_power_export)
#************************************************************************************
#                            Constraint
#************************************************************************************
#
def rule_imported_power(model, time):
  return model.VarP_Op_imp_t[time] == model.Param_P_MaxImp_t[time] * model.X_imp_t[time]
model._contracted_power_constraint_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_imported_power)

def rule_exported_power(model, time):
  return model.VarP_Op_Exp_t[time] == model.Param_P_MaxExp_t[time] * model.X_exp_t[time]
model.exported_power_constraint = pyo.Constraint(model.SetTimeIntervals, rule=rule_exported_power)

def rule_importing_exporting(model,time):
  return model.X_imp_t[time] + model.X_exp_t[time]  <= 1
model.importing_exporting_constraint= pyo.Constraint(model.SetTimeIntervals, rule =rule_importing_exporting)

def rule_balance_power(model, time, w):
  grid_Balance = (model.VarP_Op_imp_t[time] - model.VarP_Op_Exp_t[time] + model.Var_r_Up_imp_t_w[time, w] - model.Var_r_Down_imp_t_w[time, w])
  EC_Balance = (
          sum(model.Param_Pld_l_t[load, time] for load in model.LoadSet) +
          sum(model.VarPstorageCharge_b_t[s, time, w] - model.VarPstorageDischarge_b_t[s, time, w] for s in model.BatterySet) +
          sum(model.VarP_OpGen_g_t[gen, time] + model.Var_r_UpGen_g_t_w[gen, time, w] - model.Var_r_DownGen_g_t_w[gen, time, w] for gen in model.GenSet)
  )
  return grid_Balance == EC_Balance
model.balance_power_constraint = pyo.Constraint(model.SetTimeIntervals, model.SetSceneries, rule=rule_balance_power)

def rule_PstorageCharge(model, s, time, w):
  return model.VarPstorageCharge_b_t[s, time, w] <= model.ParamPstorageChargeLimit_b_t[s, time] * model.XstorageCharge_b_t[s, time, w]
model.PstorageCharge_constraint = pyo.Constraint(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, rule=rule_PstorageCharge)

def rule_PstorageDischarge(model, s, time, w):
  return model.VarPstorageDischarge_b_t[s, time, w] <= model.ParamPstorageDischargeLimit_b_t[s, time] * model.XstorageDischarge_b_t[s, time, w]
model.PstorageDischarge_constraint = pyo.Constraint(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, rule=rule_PstorageDischarge)

def rule_battery_status(model, s, time, w):
  return model.XstorageDischarge_b_t[s, time, w] + model.XstorageCharge_b_t[s, time, w] <= 1
model.battery_status_constraint = pyo.Constraint(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, rule=rule_battery_status)

def rule_Inicial_Energy_Storage(model, s, time, w):
  ChargeEfficiency = StorageCharacteritics_dict["8 Charge Efficiency (%)"][s] / 100
  DischargeEfficiency = 100 / StorageCharacteritics_dict["9 Discharge Efficiency (%)"][s]
  if time == 't_0':
    SoC = StorageCharacteritics_dict["10 Initial State (%)"][s]/100
    return model.VarEnergyStorage_b_t[s, time, w] == model.EstorageMax_b[s] * SoC + (model.VarPstorageCharge_b_t[s, time, w] * ChargeEfficiency) - (model.VarPstorageDischarge_b_t[s, time, w] * DischargeEfficiency)
  else:
    return model.VarEnergyStorage_b_t[s, time, w] == model.VarEnergyStorage_b_t[s, model.SetTimeIntervals.prev(time), w] + (model.VarPstorageCharge_b_t[s, time, w] * ChargeEfficiency) - (model.VarPstorageDischarge_b_t[s, time, w] * DischargeEfficiency)
model.Inicial_Energy_Storage_constraint = pyo.Constraint(model.BatterySet, model.SetTimeIntervals, model.SetSceneries, rule=rule_Inicial_Energy_Storage)


def rule_Inicial_Energy_V2G(model, v2g, time, w):
  if model.Ev2gInitial_v_t[v2g, time] != 0:
    return model.VarEnergyV2G_v_t[v2g, time, w] == model.Ev2gInitial_v_t[v2g, time]
  else:
    return pyo.Constraint.Skip
model.EnergyInitial_V2G_constraint = pyo.Constraint(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, rule=rule_Inicial_Energy_V2G)

def rule_rolling_Energy_V2G(model, v2g, time, w):
  if model.Ev2gInitial_v_t[v2g, time] == 0 and model.XconexionStatus_v_t[v2g, time, w] == 1:
    return model.VarEnergyV2G_v_t[v2g, time, w] == model.VarEnergyV2G_v_t[v2g, model.SetTimeIntervals.prev(time), w] + model.VarPv2gCharge_v_t[v2g, time, w] - model.VarPv2gDischarge_v_t[v2g, time, w]
  else:
    return pyo.Constraint.Skip
model.rolling_Energy_V2G_constraint = pyo.Constraint(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, rule=rule_rolling_Energy_V2G)

def rule_ConexionStatus_Energy_V2G(model, v2g, time, w):
  if model.XconexionStatus_v_t[v2g, time, w] == 0:
    return model.VarEnergyV2G_v_t[v2g, time, w] == 0
  else:
    return pyo.Constraint.Skip
model.ConexionStatus_Energy_V2G_constraint = pyo.Constraint(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, rule=rule_ConexionStatus_Energy_V2G)

def rule_Exit_Energy_V2G(model, v2g, time, w):
  if model.ParamEv2gExitRequired_v_t[v2g, time, w] != 0 and model.XconexionStatus_v_t[v2g, time, w] == 1:
    return model.VarEnergyV2G_v_t[v2g, time, w] >= model.ParamEv2gExitRequired_v_t[v2g, time, w]
  else:
    return pyo.Constraint.Skip
model.Exit_Energy_V2G_constraint = pyo.Constraint(model.V2G_Set, model.SetTimeIntervals, model.SetSceneries, rule=rule_Exit_Energy_V2G)
#********************************************************************************************************
                                 # Objective Function Minimization of Cost
#********************************************************************************************************
def objective_rule(model):
  F_DA = sum(
    model.VarR_UpGen_g_t[gen, time] * model.ParamC_UpGenCost_g_t[gen, time] + model.VarR_DownGen_g_t[gen, time] * model.ParamC_DownGenCost_g_t[gen, time] for gen in model.GenSet for time in
    model.SetTimeIntervals)
  F_DA += sum(model.VarR_UpImp_t[time] * model.Param_C_UpImpPrice_t[time] + model.VarR_DownImp_t[time] * model.Param_C_DownImpPrice_t[time] for time in model.SetTimeIntervals)

  F_RT = sum(pi_w[w] * (
          sum(model.Var_r_UpGen_g_t_w[gen, time, w] * model.Param_c_Gen_Up_g_t[gen, time] +
              model.Var_r_DownGen_g_t_w[gen, time, w] * model.Param_c_Gen_Down_g_t[gen, time] +
              model.VarP_OpGen_g_t[gen, time] * model.Param_c_Gent_g_t[gen, time] for gen in model.GenSet for time in
              model.SetTimeIntervals)
          + sum(model.Var_r_Up_imp_t_w[time, w] * model.Param_c_UpImp_t[time] +
                model.Var_r_Down_imp_t_w[time, w] * model.Param_c_DownImp_t[time] +
                model.VarP_Op_imp_t[time] * model.Param_C_ImpPrice_t[time] -
                model.VarP_Op_Exp_t[time] * model.Param_C_ExpPrice_t[time] for time in model.SetTimeIntervals)
          + sum(
    model.VarPstorageCharge_b_t[s, time, w] * model.Param_c_St_Ch_s_t[s, time] + model.VarPstorageDischarge_b_t[
      s, time, w] * model.Param_c_St_Dch_s_t[s, time] +
    model.VarP_St_minRlx_s_t_w[s, time, w] * model.Param_c_St_minRlx_s_t[s, time] for s in model.BatterySet for time in
    model.SetTimeIntervals)
          + sum(
    model.VarPv2gDischarge_v_t[v2g, time, w] * model.Param_c_EV_Dch_e_t[v2g, time] - model.VarPv2gCharge_v_t[
      v2g, time, w] * model.Param_c_EV_Ch_e_t[v2g, time]  for v2g in model.V2G_Set for time in
    model.SetTimeIntervals)
  ) for w in model.SetSceneries)
  return F_DA + F_RT

model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
solver = pyo.SolverFactory('cplex', executable=cplex_path)
solver.set_executable(cplex_path, validate=False)
results = solver.solve(model, tee=True)
print(results)


def ext_pyomo_vals(vals):
  # Create a pandas Series from the Pyomo values
  s = pd.Series(vals.extract_values(),index=vals.extract_values().keys())
  # Check if the Series is multi-indexed, if so, unstack it
  if type(s.index[0]) == tuple:  # it is multi-indexed
    s = s.unstack(level=1)
  else:
    # Convert Series to DataFrame
    s = pd.DataFrame(s)
  return s

import matplotlib.pyplot as plt
df_VarP_OpGen_g_t = ext_pyomo_vals(model.VarP_OpGen_g_t)
df_VarR_UpGen_g_t = ext_pyomo_vals(model.VarR_UpGen_g_t)
df_VarR_DownGen_g_t = ext_pyomo_vals(model.VarR_DownGen_g_t)
df_VarR_UpImp_t = ext_pyomo_vals(model.VarR_UpImp_t)
df_VarR_DownImp_t = ext_pyomo_vals(model.VarR_DownImp_t)
df_Var_r_UpGen_g_t_w = ext_pyomo_vals(model.Var_r_UpGen_g_t_w)
df_Var_r_DownGen_g_t_w = ext_pyomo_vals(model.Var_r_DownGen_g_t_w)
df_VarP_Op_imp_t = ext_pyomo_vals(model.VarP_Op_imp_t)
df_Var_r_Up_imp_t_w = ext_pyomo_vals(model.Var_r_Up_imp_t_w)
df_Var_r_Down_imp_t_w = ext_pyomo_vals(model.Var_r_Down_imp_t_w)
df_VarP_Op_Exp_t = ext_pyomo_vals(model.VarP_Op_Exp_t)
df_VarPstorageCharge_b_t = ext_pyomo_vals(model.VarPstorageCharge_b_t)
df_VarPstorageDischarge_b_t = ext_pyomo_vals(model.VarPstorageDischarge_b_t)

def plot_profiles(varP_OpGen, varR_UpGen, varR_DownGen, varP_Op_imp, varP_Op_Exp, Pload_Forecast, VarPstorageCharge,
                  VarPstorageDischarge):
  # Sort the index based on the numeric part of the time labels
  def sort_series(series):
    return series.reindex(sorted(series.index, key=lambda x: int(x.split('_')[1])))

  varP_OpGen = sort_series(varP_OpGen)
  varR_UpGen = sort_series(varR_UpGen)
  varR_DownGen = sort_series(varR_DownGen)
  varP_Op_imp = sort_series(varP_Op_imp)
  varP_Op_Exp = sort_series(varP_Op_Exp)
  Pload_Forecast = sort_series(Pload_Forecast)
  VarPstorageCharge = sort_series(VarPstorageCharge)
  VarPstorageDischarge = sort_series(VarPstorageDischarge)

  plt.figure(figsize=(14, 8))

  # Plot each variable
  plt.plot(varP_OpGen, label=r"$P_{Gen(g, t)}^{Op}$", marker='o')
  plt.plot(varR_UpGen, label=r"$R_{Gen(g, t)}^{UP}$", marker='x')
  plt.plot(varR_DownGen, label=r"$R_{Gen(g, t)}^{Down}$", marker='s')
  plt.plot(varP_Op_imp, label=r"$P_{\operatorname{Imp}(t)}^{Op}$", marker='^')
  plt.plot(varP_Op_Exp, label=r"$P_{\operatorname{Exp}(t)}^{Op}$", marker='d')
  plt.plot(Pload_Forecast, label=r"$P_{Ld(l, t, w)}$", marker='*', linestyle='--')
  plt.plot(VarPstorageCharge, label=r"$P_{St(s, t, w)}^{ch}$", marker='v', linestyle='-.')
  plt.plot(VarPstorageDischarge, label=r"$P_{St(s, t, w)}^{dis}$", marker='p', linestyle=':')

  # Add titles and labels
  plt.title('Profiles of Generation, Reserves, Load Forecast, and Storage Operations Over Time')
  plt.xlabel('Time (t)')
  plt.ylabel('Values')
  plt.legend()
  plt.grid(True)

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)

  # Show the plot
  plt.show()


plot_profiles(df_VarP_OpGen_g_t.sum(axis=0), df_VarR_UpGen_g_t.sum(axis=0), df_VarR_DownGen_g_t.sum(axis=0),
              df_VarP_Op_imp_t, df_VarP_Op_Exp_t, df_Pload_Forecast.sum(axis=0), df_VarPstorageCharge_b_t.sum(axis=0),
              df_VarPstorageDischarge_b_t.sum(axis=0))
